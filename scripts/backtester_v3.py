"""バックテスタ v3 — これまでに確立した施策を両側同時シミュレーションへ統合する。

【なぜ v3 が要るか】
  報告 23〜31 の分析はすべて**片側の約定 1 件ずつ**を数える観察研究だった。
  そのため 2 つの問題が残っていた。
    ・**手仕舞いコストが未解決** … 中値 +0.2303 / メイカー +0.1423 / テイカー −0.4137
      と前提次第で符号が変わる
    ・**再参入を数えていない** … 引いた後に出し直して別の約定を取る機会が未計上
  両側同時に置いて往復とも約定すれば、**スプレッドを 2 回受け取る**ので
  手仕舞いコストは構造的に消える。再参入も自然に入る。

【統合する 4 つの施策】
  A 気配改善     … スプレッドが広いとき最良気配の内側 1 ティックに置く
                   (gate_optimization_report.md で唯一確立: +1,521〜2,162 bp/日)
  B 在庫制約     … 建玉の絶対値が上限を超えたら増やす側は出さない(同報告で必須)
  C 前方約定フロー … 前で 10ms 以内に 5 単位超が約定したら引く
                   (pull_rule_99_report.md: +25.5 bp/日、両体制×両サイド)
  D 価格予測ゲート … L10 book_slope のスプライン予測が不利な側は出さない
                   (slope_spline_report.md: 1 秒地平で標本外 R² 0.0723)

【★時間契約とルックアヘッド】
  ・スプライン模型は**バックテスト期間より前の 20 日だけ**で当てはめたものを使う
  ・シグナルは常に δ=100ms 前の値で判断する(反応時間の予算)
  ・板・約定はその時刻までのイベントのみ
  ・C の判定に使う前方約定フローは、自分より前の注文の約定のみ。
    自分は発注時点で最後尾なので、**その価格帯の約定はすべて自分より前**である

【損益】
  現金 + 建玉評価(日終の中値)。メイカー手数料 0.088 bp を約定ごとに控除。
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.interpolate import BSpline

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
LIFE = Path("data/l2_v99/lifecycle")
SP = D / "slope_spline"
FEE_BP = 0.088
LAT = 100_000_000                 # 反応時間 δ = 100ms
QTY = 1.0
TICK = 0.001
NON_RESTING = ["Ioc", "FrontendMarket", "LiquidationMarket"]
PULL_WIN = 10_000_000             # 前方約定フローの窓 10ms
PULL_VOL = 5.0                    # 閾値 5 単位


def load_model():
    j = json.loads((D / "slope_spline.json").read_text(encoding="utf-8"))
    m = j["model_for_backtest"]
    return (np.array(m["knots_bid"]), np.array(m["knots_ask"]),
            np.array(m["coef"]), m["deg"], m["train_days"])


def predict(sb, sa, tb, ta, coef, deg):
    """スプライン模型による 1 秒先リターンの予測 [bp]。"""
    def bas(x, t):
        return BSpline.design_matrix(np.clip(x, t[deg], t[-deg - 1]), t, deg).toarray()[:, :-1]
    nb = len(tb) - deg - 2
    return coef[0] + bas(sb, tb) @ coef[1:1 + nb] + bas(sa, ta) @ coef[1 + nb:]


def load(dt, model):
    f = SP / f"{dt}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    if "ts" not in z.files:
        return None
    tb, ta, coef, deg, _ = model
    sig_ts = z["ts"]
    sig = predict(z["sb"].astype(np.float64), z["sa"].astype(np.float64), tb, ta, coef, deg)

    life = (pl.read_parquet(LIFE / f"dt={dt}" / "part-000.parquet",
                            columns=["side", "px", "ts_open", "ts_close",
                                     "tif", "is_rejected", "is_trigger"])
            .filter((~pl.col("is_rejected")) & (~pl.col("is_trigger"))
                    & (~pl.col("tif").is_in(NON_RESTING).fill_null(False))
                    & pl.col("ts_open").is_not_null() & pl.col("ts_close").is_not_null()))
    fs = sorted(glob.glob(f"data/fills_v99/dt={dt}/**/*.parquet", recursive=True))
    fl = pl.concat([pl.read_parquet(x, columns=["ts", "px", "sz", "side", "crossed", "tid"])
                    for x in fs], how="diagonal_relaxed")
    # ★決定性: 同一 ms の fills の順序を tid で固定(capacity_v3.load と同じ理由)
    fill = (fl.filter(pl.col("crossed")).unique(subset=["tid"], keep="first",
                                                maintain_order=True)
              .with_columns(pl.col("ts").cast(pl.Int64).alias("t")).sort(["t", "tid"]))
    mp = (pl.read_parquet(D / f"microprice/dt={dt}/part-000.parquet",
                          columns=["ts", "best_bid", "best_ask", "mid", "is_crossed"])
          .filter(~pl.col("is_crossed")).sort("ts"))
    # 価格レベルごとに (到着時刻昇順, ts_close の前置最大値, 到着時の残量累積)
    px = life["px"].to_numpy(); sd = life["side"].to_numpy()
    o = life["ts_open"].to_numpy(); c = life["ts_close"].to_numpy()
    key = np.round(px * 1e6).astype(np.int64) * 2 + (sd == "B")
    tmp: dict[int, list] = {}
    for k, oo, cc in zip(key, o, c):
        tmp.setdefault(int(k), []).append((int(oo), int(cc)))
    lvl = {}
    for k, v in tmp.items():
        v.sort()
        lvl[k] = (np.array([x[0] for x in v], dtype=np.int64),
                  np.maximum.accumulate(np.array([x[1] for x in v], dtype=np.int64)))
    # 価格レベルごとの約定フロー(時刻, 数量の累積)。C の判定に使う
    fpx = fill["px"].to_numpy(); fsd = fill["side"].to_numpy()
    fts = fill["t"].to_numpy(); fsz = fill["sz"].to_numpy()
    fkey = np.round(fpx * 1e6).astype(np.int64) * 2 + (fsd == "A")   # テイカー売り→買い板
    flow = {}
    for k in np.unique(fkey):
        m = fkey == k
        t_ = fts[m]; s_ = fsz[m]
        oo = np.argsort(t_)
        flow[int(k)] = (t_[oo], np.concatenate([[0.0], np.cumsum(s_[oo])]))
    return {"lvl": lvl, "flow": flow, "fill": fill, "mp": mp,
            "sig_ts": sig_ts, "sig": sig}


def simulate(d, use_improve, inv_limit, use_pull, use_gate, gate_thr):
    mp = d["mp"]
    mts = mp["ts"].to_numpy()
    bb = mp["best_bid"].to_numpy(); ba = mp["best_ask"].to_numpy()
    mid = mp["mid"].to_numpy()
    fl = d["fill"]
    f_t = fl["t"].to_numpy(); f_px = fl["px"].to_numpy(); f_side = fl["side"].to_numpy()
    trades = {s: (f_t[f_side == ("A" if s == "B" else "B")],
                  f_px[f_side == ("A" if s == "B" else "B")]) for s in ("B", "A")}
    sig_ts, sig = d["sig_ts"], d["sig"]

    intervals = []
    for side, ser in (("B", bb), ("A", ba)):
        i = 0; n = mts.size
        while i < n:
            p = ser[i]; j = i
            while j + 1 < n and ser[j + 1] == p:
                j += 1
            intervals.append((mts[i], mts[j + 1] if j + 1 < n else mts[-1], side, p, i))
            i = j + 1
    intervals.sort()

    cash = 0.0; pos = 0.0
    fills = []; pulled = 0; quoted = 0
    for t0, t_end, side, p, i0 in intervals:
        s_dir = 1.0 if side == "B" else -1.0
        if inv_limit > 0 and s_dir * pos >= inv_limit:
            continue
        # D 価格予測ゲート: δ 前の予測が自分に不利なら出さない
        if use_gate:
            k = int(np.searchsorted(sig_ts, t0 - LAT, side="right")) - 1
            if k < 0 or s_dir * sig[k] < gate_thr:
                continue
        px_ = p
        if use_improve and (ba[i0] - bb[i0]) > 2 * TICK * 1.5:
            px_ = p + TICK if side == "B" else p - TICK
        quoted += 1
        k_ = int(round(px_ * 1e6)) * 2 + (side == "B")
        lv = d["lvl"].get(k_)
        t_front = t0
        if px_ == p and lv is not None:
            jj = int(np.searchsorted(lv[0], t0, side="right")) - 1
            if jj >= 0 and lv[1][jj] > t0:
                t_front = int(lv[1][jj])
        tt, tp = trades[side]
        lo = np.searchsorted(tt, max(t0, t_front)); hi = np.searchsorted(tt, t_end)
        got = None
        for x in range(lo, hi):
            if abs(tp[x] - px_) <= 1e-9:
                got = int(tt[x]); break
        if got is None:
            continue
        # C 前方約定フロー: 約定より δ 以上前に閾値を超えたら引く
        if use_pull:
            fw = d["flow"].get(k_)
            if fw is not None:
                ft, fc = fw
                a = np.searchsorted(ft, t0, side="right")
                b = np.searchsorted(ft, got - LAT, side="right")
                if b > a:
                    lo2 = np.searchsorted(ft, ft[a:b] - PULL_WIN, side="left")
                    win = fc[a + 1:b + 1] - fc[lo2]
                    if win.size and win.max() > PULL_VOL:
                        pulled += 1
                        continue
        cash += -s_dir * px_ * QTY - FEE_BP / 1e4 * px_ * QTY
        pos += s_dir * QTY
        # 分解用: 約定時点の中値からの粗利と、1 秒後までの逆選択
        j0 = int(np.searchsorted(mts, got, side="right")) - 1
        j1 = int(np.searchsorted(mts, got + 1_000_000_000, side="right")) - 1
        m0 = mid[max(j0, 0)]
        gr = s_dir * (m0 - px_) / m0 * 1e4
        ad = (s_dir * (mid[max(j1, 0)] - m0) / m0 * 1e4
              if j1 >= 0 and got + 1_000_000_000 <= mts[-1] else np.nan)
        fills.append((got, s_dir, px_, gr, ad))
    if not fills:
        return None
    pnl = cash + pos * mid[-1]
    notional = float(np.mean([abs(f[2]) for f in fills])) * QTY
    gr = np.array([f[3] for f in fills])
    ad = np.array([f[4] for f in fills])
    ad = ad[np.isfinite(ad)]
    return {"n_quote": quoted, "n_fill": len(fills), "n_pull": pulled,
            "pnl_bp": float(pnl / notional * 1e4),
            "per_fill": float(pnl / notional * 1e4 / len(fills)),
            "gross_bp": float(gr.mean()),
            "adv_bp": float(ad.mean()) if ad.size else float("nan"),
            "buy_frac": float(np.mean([f[1] > 0 for f in fills])),
            "end_inv": float(pos)}


def main() -> None:
    model = load_model()
    days = sys.argv[1:] or sorted(p.stem for p in SP.glob("*.npz"))
    days = [d for d in days if d not in set(model[4])]      # 学習日は除外
    # ★閾値 0 は「予測が少しでも不利なら出さない」で約半分を止める。
    #   両側マーケットメイクでは片側を止めると往復のスプレッド獲得を失うので、
    #   緩い閾値(強く不利なときだけ止める)も試す。
    POL = [("A+B 基準", True, 5.0, False, False, 0.0),
           ("A+B+C 前方フロー", True, 5.0, True, False, 0.0),
           ("A+B+D ゲート0", True, 5.0, False, True, 0.0),
           ("A+B+D ゲート-0.1", True, 5.0, False, True, -0.1),
           ("A+B+D ゲート-0.3", True, 5.0, False, True, -0.3),
           ("A+B+D ゲート-1.0", True, 5.0, False, True, -1.0),
           ("A+B+C+D ゲート-0.3", True, 5.0, True, True, -0.3)]
    rows = []
    for dt in days:
        d = load(dt, model)
        if d is None:
            continue
        for nm, imp, inv, pu, ga, th in POL:
            r = simulate(d, imp, inv, pu, ga, th)
            if r:
                rows.append({"dt": dt, "policy": nm, **r})
        print(f"{dt} 済", flush=True)
        pl.DataFrame(rows).write_csv(D / "backtest_v3.csv")
    pl.DataFrame(rows).write_csv(D / "backtest_v3.csv")


if __name__ == "__main__":
    main()
