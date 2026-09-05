"""候補テーブルのルックアヘッド監査。

主張(「X_t は t までに観測可能な値だけ」)をコードの読みで済ませない。
別経路で計算し直して一致を確かめ、さらに「漏れていたら気づける検定か」を
確かめるための陽性対照(わざと未来を使った特徴量)を置く。

 A. 独立再計算 — 標本行について、生の bbo / fills / L1 から明示的な
    マスク(t−w < s ≤ t)で計算し直し、格納値と突き合わせる
 B. 板の時刻 — depth_own(100ms 格子のラダー由来)が、
    「セル開始時刻の bbo」と「t 時点の bbo」のどちらに一致するか。
    先読みしているなら後者に寄る
 C. 陽性対照 — 未来 1 秒の OFI を作り、約定の判別力を比べる。
    後ろ向き版がこれに近ければ漏れを疑う
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import CANCELS, GRID_NS, RESTING, TERMINAL, clean_bbo
from plot_latency import auc

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BULK = Path("E:/Memory-quotes")
DAY_NS = 86_400_000_000_000
NS = 3000
SEED = 12345


def audit_day(tag: str, dt: str, rng: np.random.Generator) -> dict:
    T = pl.read_parquet(BULK / tag / f"dt={dt}.parquet").filter(pl.col("side") == 1)
    d = clean_bbo(pl.scan_parquet(DATA / f"bbo_{tag}.parquet")
                  .filter(pl.col("dt") == dt).collect())[0].sort("ts")
    ts = d["ts"].cast(pl.Int64).to_numpy()
    pb, pa = d["best_bid"].to_numpy(), d["best_ask"].to_numpy()
    qb1, qa1 = d["bid_sz"].to_numpy(), d["ask_sz"].to_numpy()
    d0 = int(ts[0]) // DAY_NS * DAY_NS

    F = pl.scan_parquet(DATA / f"fills_{tag}.parquet").filter(
        pl.col("crossed") & (pl.col("dt") == dt)).select(
        "ts", "sz", "side").collect().sort("ts")
    ft = F["ts"].cast(pl.Int64).to_numpy()
    fsz = F["sz"].to_numpy()
    fbuy = F["side"].to_numpy() == "B"

    E = pl.read_parquet(DATA / f"l1_{tag}/dt={dt}.parquet",
                        columns=["ts", "status", "orig_sz", "remaining_sz",
                                 "tif", "is_trigger"]).filter(
        (~pl.col("is_trigger")) & pl.col("tif").is_in(RESTING)
        & pl.col("status").is_in(["open"] + TERMINAL)).sort("ts")
    et = E["ts"].cast(pl.Int64).to_numpy()
    est = E["status"].to_numpy()
    evol = np.where(est == "open", E["orig_sz"].to_numpy(),
                    E["remaining_sz"].to_numpy())
    is_can = np.isin(est, CANCELS)

    n = T.height
    smp = np.sort(rng.choice(n, min(NS, n), replace=False))
    tq = T["ts"].cast(pl.Int64).to_numpy()[smp]

    out = {"dt": dt, "n": int(n), "n_smp": int(smp.size)}

    # ---- A. 独立再計算(明示的なマスク。cumsum を使わない) ----
    def win_sum(v, arr, t, w):
        # 窓 (t-w, t] の端だけ二分探索で決め、スライスをそのまま足す。
        # 累積和の差分とは別の演算経路なので、端のずれがあれば食い違う。
        j0 = int(np.searchsorted(arr, t - w, side="right"))
        j1 = int(np.searchsorted(arr, t, side="right"))
        return float(v[j0:j1].sum())

    got, exp = {}, {}
    got["aggr_1s"] = T["aggr_1s"].to_numpy()[smp]
    got["cancel_10s"] = T["cancel_10s"].to_numpy()[smp]
    got["add_10s"] = T["add_10s"].to_numpy()[smp]
    got["ofi_1s"] = T["ofi_1s"].to_numpy()[smp]
    got["spread_bp"] = T["spread_bp"].to_numpy()[smp]
    got["queue_ahead_qty"] = T["queue_ahead_qty"].to_numpy()[smp]

    # ★ 配列の作成は内包表記の外に出す(中に置くと標本の数だけ作り直す)
    fnet = fsz * np.where(fbuy, 1.0, -1.0)
    vcan = np.where(is_can, evol, 0.0)
    vadd = np.where(est == "open", evol, 0.0)
    exp["aggr_1s"] = np.array([win_sum(fnet, ft, t, 10 ** 9) for t in tq])
    exp["cancel_10s"] = np.array([win_sum(vcan, et, t, 10 ** 10) for t in tq])
    exp["add_10s"] = np.array([win_sum(vadd, et, t, 10 ** 10) for t in tq])
    e = np.concatenate([[0.0], (
        (pb[1:] >= pb[:-1]) * qb1[1:] - (pb[1:] <= pb[:-1]) * qb1[:-1]
        - ((pa[1:] <= pa[:-1]) * qa1[1:] - (pa[1:] >= pa[:-1]) * qa1[:-1]))])
    exp["ofi_1s"] = np.array([win_sum(e, ts, t, 10 ** 9) for t in tq])
    jt = np.searchsorted(ts, tq, side="right") - 1
    exp["spread_bp"] = (pa[jt] - pb[jt]) / (0.5 * (pa[jt] + pb[jt])) * 1e4
    exp["queue_ahead_qty"] = qb1[jt]

    print(f"\n[{dt}] A. 独立再計算(標本 {smp.size:,} 行)")
    for k in got:
        dif = np.abs(np.nan_to_num(got[k]) - np.nan_to_num(exp[k]))
        sc = max(1e-12, float(np.nanmax(np.abs(exp[k]))))
        out[f"A_{k}"] = float(dif.max() / sc)
        print(f"  {k:18s} 最大差 {dif.max():.6g}  (相対 {dif.max()/sc:.2e})  "
              f"{'一致' if dif.max()/sc < 1e-5 else '★不一致'}")

    # ---- B. 板の時刻 ----
    cell = (tq - d0) // GRID_NS
    tcell = d0 + cell * GRID_NS                       # セル開始時刻
    jc = np.searchsorted(ts, tcell, side="left") - 1  # セル開始「未満」の bbo
    ok = jc >= 0
    dep = T["depth_own"].to_numpy()[smp]   # bbo の bid_sz と同じ単位
    tol = 1e-3                                    # float32 で保存しているため
    m_cell = np.abs(dep[ok] - qb1[jc[ok]]) < tol
    m_now = np.abs(dep[ok] - qb1[jt[ok]]) < tol
    diff = np.abs(qb1[jc[ok]] - qb1[jt[ok]]) >= tol   # 2 者が違う行だけが識別力を持つ
    out["B_ndiff"] = int(diff.sum())
    out["B_cell_d"] = float(m_cell[diff].mean()) if diff.any() else float("nan")
    out["B_now_d"] = float(m_now[diff].mean()) if diff.any() else float("nan")
    age = (tq - tcell) / 1e6
    out["B_cell"] = float(m_cell.mean())
    out["B_now"] = float(m_now.mean())
    print(f"[{dt}] B. 板の時刻 — depth_own の一致率  "
          f"セル開始時点の bbo {100*m_cell.mean():.2f}%  /  "
          f"t 時点の bbo {100*m_now.mean():.2f}%   "
          f"(古さ 中央 {np.median(age):.1f}ms 最大 {age.max():.1f}ms)")
    print(f"[{dt}]    うち 2 者が食い違う {int(diff.sum()):,} 行に限ると  "
          f"セル開始 {100*out['B_cell_d']:.2f}%  /  t 時点 {100*out['B_now_d']:.2f}%")

    # ---- C. 陽性対照(わざと未来を使う) ----
    cofi = np.cumsum(e)
    jf = np.clip(np.searchsorted(ts, tq + 10 ** 9, side="right") - 1, 0, ts.size - 1)
    jn = np.clip(np.searchsorted(ts, tq, side="right") - 1, 0, ts.size - 1)
    fut = cofi[jf] - cofi[jn]                          # 未来 1 秒の OFI
    y = T["label_filled_60s"].to_numpy()[smp].astype(float)
    v = np.isfinite(y) & np.isfinite(fut) & np.isfinite(got["ofi_1s"])
    out["C_back"] = float(auc(got["ofi_1s"][v], y[v]))
    out["C_fut"] = float(auc(fut[v], y[v]))
    # 未来 1 秒の OFI は約定の予測力自体が乏しく、対照として弱い。
    # 買い候補を約定させるのは「今後 60 秒の売りテイカー流量」そのものなので、
    # これを漏らした場合にどれだけ AUC が跳ねるかを検出力の目安にする。
    csell = np.concatenate([[0.0], np.cumsum(np.where(~fbuy, fsz, 0.0))])
    j60 = np.searchsorted(ft, tq + 60 * 10 ** 9, side="right")
    j00 = np.searchsorted(ft, tq, side="right")
    leak = csell[j60] - csell[j00]
    out["C_leak"] = float(auc(leak[v], y[v]))
    out["C_q"] = float(auc(-T["queue_ahead_qty"].to_numpy()[smp][v], y[v]))
    print(f"[{dt}] C. 陽性対照 — 約定の判別 AUC  "
          f"後ろ向き ofi_1s {out['C_back']:.4f}  /  "
          f"待ち行列(符号反転) {out['C_q']:.4f}  /  "
          f"★未来 60 秒の売りテイカー流量(意図的な漏れ) {out['C_leak']:.4f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    ap.add_argument("--days", nargs="*", default=None)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    fs = sorted((BULK / tag).glob("dt=*.parquet"))
    days = a.days or [fs[3].stem.split("=")[1], fs[len(fs) // 2].stem.split("=")[1],
                      fs[-4].stem.split("=")[1]]
    rng = np.random.default_rng(SEED)
    res = [audit_day(tag, dt, rng) for dt in days]
    pl.DataFrame(res).write_parquet(DATA / f"quotes_audit_{tag}.parquet")
    print("\n書き出し", DATA / f"quotes_audit_{tag}.parquet")
