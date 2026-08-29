"""監査: microprice の「グリッド行」(ts の下 9 桁が .999999999)が結論を汚していないか。

【発見の経緯】
  SDE の σ を較正するため 99 日の実現ボラティリティを測ったところ、
  2026-08-09 は 30 秒 RV 1.11%(最低水準)なのに 1 秒 RV 95.8% という
  ありえない組み合わせになった。中値の経路を直接見ると、mid<40 のイベントは
  **06:35:52.999999999 の 1 行だけ**で、その前後は 51.19 前後。
  約定はその日 20,225 件すべて 50.81〜51.27 の範囲で、**26 USD 近辺の約定は皆無**。

【何が問題か】
  この行は板の再構成で入る境界行(全 99 日に 4,792 行 = 0.02%)で、
  ときどき異常な価格を持つ。日次損益は日終の評価行が健全なので無傷だが、
  **日中の最小 equity(= 清算判定)は極値を取るので 1 行で壊れる**。
  報告 34 §4 の「10× は最悪日 2026-08-09 で清算」はこの 1 行が作った可能性が高い。

【検査】
  グリッド行を除いた場合と含めた場合で、eq_min と日次損益を突き合わせる。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from capacity_v3 import load                        # noqa: E402
from backtester_v3 import load_model, FEE_BP, TICK  # noqa: E402

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
CAP = 100.0
GRID_NS = 999_999_999


def sim_path(d, qty, inv_limit, drop_grid: bool):
    mp = d["mp"]
    if drop_grid:
        mp = mp.filter((pl.col("ts") % 1_000_000_000) != GRID_NS)
    mts = mp["ts"].to_numpy()
    bb = mp["best_bid"].to_numpy(); ba = mp["best_ask"].to_numpy()
    mid = mp["mid"].to_numpy()
    fl = d["fill"]; f_t = fl["t"].to_numpy(); f_px = fl["px"].to_numpy()
    f_sz = fl["sz"].to_numpy(); f_side = fl["side"].to_numpy()
    tr = {}
    for s in ("B", "A"):
        m = f_side == ("A" if s == "B" else "B")
        tr[s] = (f_t[m], f_px[m], f_sz[m])
    iv = []
    for side, ser in (("B", bb), ("A", ba)):
        i = 0; n = mts.size
        while i < n:
            p = ser[i]; j = i
            while j + 1 < n and ser[j + 1] == p:
                j += 1
            iv.append((mts[i], mts[j + 1] if j + 1 < n else mts[-1], side, p, i))
            i = j + 1
    iv.sort()
    cash = 0.0; pos = 0.0; ft = []; fc = []; fp = []
    for t0, t_end, side, p, i0 in iv:
        sd = 1.0 if side == "B" else -1.0
        if inv_limit > 0 and sd * pos >= inv_limit:
            continue
        px_ = p
        if (ba[i0] - bb[i0]) > 2 * TICK * 1.5:
            px_ = p + TICK if side == "B" else p - TICK
        k_ = int(round(px_ * 1e6)) * 2 + (side == "B")
        lv = d["lvl"].get(k_); t_front = t0
        if px_ == p and lv is not None:
            jj = int(np.searchsorted(lv[0], t0, side="right")) - 1
            if jj >= 0 and lv[1][jj] > t0:
                t_front = int(lv[1][jj])
        tt, tp, tz = tr[side]
        lo = np.searchsorted(tt, max(t0, t_front)); hi = np.searchsorted(tt, t_end)
        rem = qty
        for x in range(lo, hi):
            if abs(tp[x] - px_) > 1e-9:
                continue
            take = min(rem, float(tz[x]))
            cash += -sd * px_ * take - FEE_BP / 1e4 * px_ * take
            pos += sd * take; rem -= take
            ft.append(int(tt[x])); fc.append(cash); fp.append(pos)
            if rem <= 1e-9:
                break
    if not ft:
        return None
    ft = np.array(ft); fc = np.array(fc); fp = np.array(fp)
    idx = np.clip(np.searchsorted(mts, ft, side="left"), 0, len(mts) - 1)
    ends = np.append(idx[1:], len(mts))
    lows = []
    for a, b, c_, p_ in zip(idx, ends, fc, fp):
        if b <= a:
            continue
        seg = mid[a:b]
        lows.append(c_ + p_ * (seg.min() if p_ > 0 else seg.max()))
    return {"pnl": float(fc[-1] + fp[-1] * mid[-1]),
            "eq_min": float(min(lows)) if lows else 0.0,
            "max_pos_usd": float(np.abs(fp).max() * mid.mean()),
            "n_row": len(mts)}


def main() -> None:
    model = load_model()
    days = sorted(pl.read_csv(D / "capacity_100usd.csv")["dt"].unique().to_list())
    days = days + ["2026-08-09"]                 # 報告 34 が「最悪日」とした日
    PX = 53.7
    # 既存 liq_100usd.csv がグリッド行を含む版なので、除いた版だけ回して突き合わせる
    old = pl.read_csv(D / "liq_100usd.csv")
    rows = []
    for i, dt in enumerate(days):
        d = load(dt, model)
        if d is None:
            continue
        for L in (1, 3, 5, 10):
            q = CAP * L / (5 * PX)
            b = sim_path(d, q, 5.0 * q, drop_grid=True)
            if b:
                rows.append({"dt": dt, "lev": L, "pnl_drop": b["pnl"],
                             "eqmin_drop": b["eq_min"], "n_row_drop": b["n_row"]})
        print(f"  {i+1}/{len(days)} {dt}", flush=True)
        pl.DataFrame(rows).write_csv(D / "audit_grid_rows.csv")
    df = pl.DataFrame(rows).join(
        old.select(["dt", "lev", pl.col("pnl").alias("pnl_with"),
                    pl.col("eq_min").alias("eqmin_with")]), on=["dt", "lev"], how="left")
    df.write_csv(D / "audit_grid_rows.csv")

    ev = df.filter(pl.col("eqmin_with").is_not_null())
    print(f"\n=== 評価窓 30 日: グリッド行を除いた影響 ===")
    print(f"{'レバ':>4}{'損益 最大差':>14}{'eq_min 最悪(含)':>18}{'eq_min 最悪(除)':>18}{'清算日(除)':>14}")
    for L in (1, 3, 5, 10):
        s = ev.filter(pl.col("lev") == L)
        dp = float((s["pnl_with"] - s["pnl_drop"]).abs().max())
        print(f"{L:>3}×{dp:>14.4f}{float(s['eqmin_with'].min()):>18.1f}"
              f"{float(s['eqmin_drop'].min()):>18.1f}"
              f"{int((s['eqmin_drop'] <= -CAP).sum()):>10} / {s.height}")


if __name__ == "__main__":
    main()
