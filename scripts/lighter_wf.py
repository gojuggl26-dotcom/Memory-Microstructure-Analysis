r"""ウォークフォワード評価 — 学習→検証→テストを日単位で前進させる。

=============================================================================
要求(2026-09-08 オペレータ指示)への対応
=============================================================================
1. **時系列で 学習7日 → 検証2日 → テスト1日** を 1 日ずつ前進。
   既存期間(08-18〜09-05、私が既に全部見ている)内の結果は **探索的** と表示する。
2. 閾値・モデル・数量・決済方法を **凍結** し、新しい期間(09-06〜)で評価する
   → 凍結は `lighter_wf_report.py`、評価は `lighter_oos.py`。本体は 1 の担当。
3. **分割境界をまたぐ保有・ラベルを除外**。最大決済期間 H_MAX まで含めて管理する。
4. **日を単位にブートストラップ**。同じ日の全銘柄をまとめて再抽出する
   → 本体は日×銘柄の損益を出す。集計は `lighter_wf_report.py`。
5. **試した戦略数を記録**。全構成 × fold × 日 × 銘柄を台帳に残す。

=============================================================================
戦略の定義(4 つの自由度)
=============================================================================
  モデル   signal ∈ {単一特徴量 6 種, ridge(222 特徴量)}
           ★向き(符号)も訓練期間の相関から決める。テストは見ない
  閾値     訓練期間の |signal| の分位 q ∈ {0.90, 0.95, 0.99}
  数量     固定 1 単位。全取引で同じ = 単価と総額の両方が読める
  決済     時間決済 h ∈ {10s, 60s, 300s}

  損益[bp] = sign × (mid(t+h) − mid(t))/mid(t) × 1e4 − (spread(t) + spread(t+h))/2

  ★費用: テイカーで入って出る。入口で半スプレッド、出口で半スプレッドを払うので
    往復コストは入口と出口のスプレッドの**平均**(片方だけで代用しない)。
    Lighter の手数料率はデータから確定できない(trade の `maker_fee` は
    約定額に依存しない定数コード 28/32/36 でレートではない)。手数料 0 で計算する
    = **この評価は楽観側に倒れている**。
  ★数量制限・板の食い込み・待ち行列は考慮していない(1 単位が最良気配で必ず約定
    すると仮定)。これも楽観側。

  ★「何もしない」対照 = 損益 0。判定は「総額 > 0 か」であって
    「上位分位 − 下位分位 > 0 か」ではない。

=============================================================================
★時間契約
=============================================================================
  - 特徴量は t までの情報のみ(lighter_features.py の設計)
  - 閾値・符号・リッジ係数・標準化はすべて **訓練期間だけ** で推定
  - 検証期間は構成の選択にのみ使い、テスト期間は一度も見ない
  - パージ: 各ブロックの末尾 H_MAX 秒に入るエントリを落とす(要求 3)。
    保有が次のブロックへ食い込む標本は、どのブロックからも使わない
  - 特徴量のルックバック(最大 300s)はブロック頭で前ブロックのデータを見るが、
    これは過去を見るだけで**ラベルの漏洩ではない**
  - 重なり: 同一銘柄で保有中は新規建てしない(1 銘柄 1 ポジション)

出力: E:/Memory-lighter/wf/ledger_{SYM}.parquet
"""
from __future__ import annotations

import argparse
import gc
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lighter_features import build  # noqa: E402

SRC = Path("E:/Memory-lighter/wf")
BP = 1e4
H_MAX = 300.0
HORS = (10.0, 60.0, 300.0)
QS = (0.90, 0.95, 0.99)
SIGNALS = ["ofi_1s", "ofi_z", "obi_W5", "gap_12_asym", "trd_sv_5s",
           "mic_delta1_bp", "ridge"]
RIDGE_LAM = 100.0
N_TRAIN, N_VAL, N_TEST = 7, 2, 1
# ★H100 は板の被覆 14.7% で市場として成立していない(データ報告 §2)。
#   結果を見る前に、データ品質だけを理由に外す。
SYMS = ["MU", "SNDK", "SKHYNIXUSD", "SAMSUNGUSD", "DRAM",
        "XAU", "XAG", "NVDA", "TSLA", "AAPL", "AMZN", "MSFT"]


def day_of(ts: np.ndarray) -> np.ndarray:
    return (ts // 86400).astype(np.int64)


def day_str(d: int) -> str:
    return datetime.fromtimestamp(int(d) * 86400, tz=timezone.utc).strftime("%Y-%m-%d")


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    G = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(G, X.T @ y)


def prep(sym: str):
    """特徴量行列(float32)・目的変数・往復費用・日付を返す。

    ★RAM 15GB のマシンで 1 銘柄 180 万行 × 222 列を扱うので、
      float64 の全体行列は作らない(実際に OOM で落ちた実績がある)。
    """
    d = pl.read_parquet(SRC / f"grid_{sym}.parquet")
    FT = build(d)
    del d
    gc.collect()
    ts = FT["ts_s"].to_numpy().astype(np.float64)
    bad = FT["_bad"].to_numpy()
    mid = np.where(bad, np.nan, FT["_mid"].to_numpy().astype(np.float64))
    cols = [c for c in FT.columns if not c.startswith("_") and c != "ts_s"]
    X = np.empty((len(ts), len(cols)), dtype=np.float32)
    for i, c in enumerate(cols):
        v = FT[c].to_numpy()
        X[:, i] = np.where(bad, np.nan, v)
    spr = X[:, cols.index("spr_bp")].astype(np.float64)
    del FT
    gc.collect()
    Y, COST = {}, {}
    for h in HORS:
        j = np.searchsorted(ts, ts + h, side="right") - 1
        ok = (ts[j] >= ts + h - 2) & (ts[j] > ts) & np.isfinite(mid[j]) & np.isfinite(mid)
        Y[h] = np.where(ok, (mid[j] - mid) / mid * BP, np.nan)
        # 往復コスト = 入口と出口の実測スプレッドの平均(半スプレッド × 2)
        COST[h] = np.where(ok, (spr + spr[j]) / 2.0, np.nan)
    return ts, day_of(ts), X, cols, Y, COST


def block_mask(days: np.ndarray, ts: np.ndarray, d0: int, d1: int) -> np.ndarray:
    """[d0, d1] 日のブロック。★末尾 H_MAX 秒のエントリはパージ(要求 3)。

    保有(最大 H_MAX 秒)がブロックの外へ食い込む標本を、どのブロックからも使わない。
    """
    m = (days >= d0) & (days <= d1)
    if not m.any():
        return m
    return m & (ts + H_MAX <= (d1 + 1) * 86400.0)


def one_position(fire: np.ndarray, ts: np.ndarray, h: float) -> np.ndarray:
    """保有中は新規建てしない(1 銘柄 1 ポジション)。"""
    idx = np.flatnonzero(fire)
    out = np.zeros(len(fire), dtype=bool)
    free_at = -1e18
    for i in idx:
        t = ts[i]
        if t >= free_at:
            out[i] = True
            free_at = t + h
    return out


def evaluate(sig, thr, sgn, ts, days, Y, COST, h, blk):
    """構成の損益を日別に集計。

    ★粗利と費用を分けて返す。費用(スプレッド)が結論を決めてしまう場面で、
      「信号に価値が無い」のか「費用が高すぎる」のかを区別できないと読めない。
      返り値 {day_int: (純損益, 取引数, 粗利, 費用)}
    """
    y, c = Y[h], COST[h]
    ok = blk & np.isfinite(sig) & np.isfinite(y) & np.isfinite(c)
    fire = ok & (np.abs(sig) >= thr)
    if not fire.any():
        return {}, 0
    fire = one_position(fire, ts, h)
    if not fire.any():
        return {}, 0
    gross = np.sign(sig[fire]) * sgn * y[fire]
    cost = c[fire]
    pnl = gross - cost
    dd = days[fire]
    out = {}
    for d in np.unique(dd):
        s = dd == d
        out[int(d)] = (float(pnl[s].sum()), int(s.sum()),
                       float(gross[s].sum()), float(cost[s].sum()))
    return out, int(fire.sum())


def run_symbol(sym: str) -> pl.DataFrame:
    ts, days, X, cols, Y, COST = prep(sym)
    ci = {c: i for i, c in enumerate(cols)}
    uday = np.array(sorted(set(days.tolist())))
    uday = np.array([d for d in uday
                     if np.isfinite(X[days == d, ci["spr_bp"]]).sum() > 3600])
    need = N_TRAIN + N_VAL + N_TEST
    ledger = []
    if len(uday) < need:
        print(f"{sym}: 使える日 {len(uday)} < {need}、飛ばす", flush=True)
        return pl.DataFrame(ledger)
    keep = [i for i in range(X.shape[1]) if np.isfinite(X[:, i]).mean() > 0.8]
    print(f"{sym}: {len(ts):,} 行 / 日 {len(uday)} "
          f"({day_str(uday[0])}..{day_str(uday[-1])}) / リッジ列 {len(keep)}",
          flush=True)

    for k in range(len(uday) - need + 1):
        tr = uday[k:k + N_TRAIN]
        va = uday[k + N_TRAIN:k + N_TRAIN + N_VAL]
        te = int(uday[k + N_TRAIN + N_VAL])
        m_tr = block_mask(days, ts, tr[0], tr[-1])
        m_va = block_mask(days, ts, va[0], va[-1])
        m_te = block_mask(days, ts, te, te)
        if m_tr.sum() < 50000 or m_va.sum() < 10000 or m_te.sum() < 5000:
            continue
        i_tr = np.flatnonzero(m_tr)
        # --- 訓練内で決める: 標準化・リッジ係数 ---
        Xtr = X[np.ix_(i_tr, keep)].astype(np.float64)
        med = np.nanmedian(Xtr, axis=0)
        iqr = (np.nanpercentile(Xtr, 75, axis=0)
               - np.nanpercentile(Xtr, 25, axis=0))
        iqr = np.where(iqr > 1e-12, iqr, np.inf)
        Ztr = np.clip(np.nan_to_num((Xtr - med) / iqr, nan=0.0), -5, 5)
        del Xtr
        ytr = Y[60.0][m_tr]
        g = np.isfinite(ytr)
        ridge_sig = None
        if g.sum() > 5000:
            w = ridge_fit(Ztr[g], ytr[g], RIDGE_LAM)
            ridge_sig = np.full(len(ts), np.nan)
            ridge_sig[i_tr] = Ztr @ w
            for mm in (m_va, m_te):
                i_ = np.flatnonzero(mm)
                Zb = np.clip(np.nan_to_num(
                    (X[np.ix_(i_, keep)].astype(np.float64) - med) / iqr,
                    nan=0.0), -5, 5)
                ridge_sig[i_] = Zb @ w
                del Zb
        del Ztr
        gc.collect()

        for sname in SIGNALS:
            if sname == "ridge":
                if ridge_sig is None:
                    continue
                sig = ridge_sig
            else:
                sig = X[:, ci[sname]].astype(np.float64)
            v = sig[m_tr]
            gg = np.isfinite(v) & np.isfinite(ytr)
            if gg.sum() < 5000:
                continue
            c = np.corrcoef(v[gg], ytr[gg])[0, 1]
            if not np.isfinite(c) or c == 0:
                continue
            sgn = float(np.sign(c))          # ★向きも訓練内で決める
            av = np.abs(v[np.isfinite(v)])
            for q in QS:
                thr = float(np.quantile(av, q))
                for h in HORS:
                    for tag, mm in (("val", m_va), ("test", m_te)):
                        res, _ = evaluate(sig, thr, sgn, ts, days, Y, COST,
                                          h, mm)
                        for d, (tot, n, gro, cst) in res.items():
                            ledger.append({
                                "symbol": sym, "fold": k, "split": tag,
                                "signal": sname, "q": q, "h": h,
                                "sign": sgn, "thr": thr,
                                "day": day_str(d), "n_trade": n,
                                "pnl_sum": tot, "pnl_mean": tot / max(n, 1),
                                "gross_sum": gro, "cost_sum": cst,
                                "test_day": day_str(te),
                                "train_from": day_str(tr[0]),
                                "train_to": day_str(tr[-1]),
                            })
        print(f"  fold {k}: 訓練 {day_str(tr[0])}..{day_str(tr[-1])} / "
              f"検証 {day_str(va[0])}..{day_str(va[-1])} / "
              f"テスト {day_str(te)}", flush=True)
    return pl.DataFrame(ledger)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(SYMS))
    a = ap.parse_args()
    for s in a.symbols.split(","):
        L = run_symbol(s)
        if L.height:
            L.write_parquet(SRC / f"ledger_{s}.parquet")
            print(f"{s}: 台帳 {L.height:,} 行", flush=True)
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
