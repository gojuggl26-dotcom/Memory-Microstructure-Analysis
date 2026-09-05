"""約定 delta ms 前の標本外 AUC と、実際の取り消し往復遅延を並べて描く。

    uv run python scripts/plot_latency.py --coin xyz:MU
出力: charts/<coin>_latency.png / data/latency_auc_<coin>.csv

【読み方】
AUC が上がる delta があっても、そこまで速く取り消せなければ意味がない。
右の遅延の図と重ねて読むこと。ブロック間隔の中央値は 131 ms で、
実測でも「同一ブロック内でない取り消し」の 50 ms 以内は 0.01% しかない。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_latency import DELTAS_MS, FEATS  # noqa: E402
from plot_vol import BASELINE, GRID, INK, INK2, MUTED, SURFACE, style  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LF = chr(10)
TRAIN_FRAC = 0.60
L2 = 1.0
FN = {"obi": "OBI", "depth": "自分側の厚み", "spread": "スプレッド",
      "ofi_1s": "OFI (1秒)", "ofi_100ms": "OFI (100ms)",
      "aggr_1s": "攻撃的流量 (1秒)", "aggr_100ms": "攻撃的流量 (100ms)",
      "qdepl": "待ち行列の消化", "drift": "T からの値動き", "age": "経過時間"}
FC = {"obi": "#7a52c9", "depth": "#1f8a5e", "spread": "#c48a12",
      "ofi_1s": "#1f5fa8", "ofi_100ms": "#7fa8dd", "aggr_1s": "#b5322f",
      "aggr_100ms": "#e08b7f", "qdepl": "#eb6834", "drift": "#9a9890",
      "age": "#c3c2b7"}


def legend(ax, fs=8, **kw):
    lg = ax.legend(fontsize=fs, frameon=True, facecolor=SURFACE, edgecolor=GRID,
                   labelcolor=INK2, **kw)
    lg.get_frame().set_linewidth(0.6)
    return lg


def auc(score, y):
    m = np.isfinite(score)
    s, yy = score[m], y[m].astype(bool)
    n1 = float(yy.sum())
    n0 = float(yy.size - n1)
    if n1 < 10 or n0 < 10:
        return np.nan
    r = np.argsort(np.argsort(s)) + 1.0
    return float((r[yy].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_logit(X, y, l2=L2):
    n, p = X.shape
    Z = np.column_stack([np.ones(n), X])

    def f(w):
        z = Z @ w
        ll = np.logaddexp(0.0, z) - y * z
        g = Z.T @ (1.0 / (1.0 + np.exp(-z)) - y)
        pen = l2 * np.concatenate([[0.0], w[1:]])
        return float(ll.sum() + 0.5 * l2 * (w[1:] ** 2).sum()), g + pen

    r = minimize(f, np.zeros(p + 1), jac=True, method="L-BFGS-B",
                 options={"maxiter": 300})
    return r.x


def wait_pct(gaps, ps):
    """任意の時点から次のブロックまでの待ち時間 (長さ重みつき) の分位点。"""
    g = np.sort(gaps[gaps > 0])
    tot = g.sum()
    out = []
    for p in ps:
        lo, hi = 0.0, float(g.max())
        for _ in range(60):
            w = 0.5 * (lo + hi)
            if np.minimum(g, w).sum() / tot < p:
                lo = w
            else:
                hi = w
        out.append(0.5 * (lo + hi))
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    D = ROOT / "data"
    R = pl.read_parquet(D / f"latency_features_{tag}.parquet")
    days = sorted(R["dt"].unique().to_list())
    ntr = int(round(len(days) * TRAIN_FRAC))
    tr_days = set(days[:ntr])
    print(f"{a.coin}: 学習 {ntr} 日 / 評価 {len(days) - ntr} 日  "
          f"標本 {R.height // len(DELTAS_MS):,} 約定")

    rows = []
    for dms in DELTAS_MS:
        S = R.filter(pl.col("delta_ms") == dms)
        istr = S["dt"].is_in(list(tr_days)).to_numpy()
        y = S["y"].to_numpy().astype(float)
        Xall = np.column_stack([S[f].to_numpy().astype(float) for f in FEATS])
        Xall = np.where(np.isfinite(Xall), Xall, np.nan)
        med = np.nanmedian(Xall[istr], axis=0)
        Xall = np.where(np.isfinite(Xall), Xall, med)
        mu = Xall[istr].mean(axis=0)
        sg = Xall[istr].std(axis=0)
        sg = np.where(sg > 0, sg, 1.0)
        Z = (Xall - mu) / sg
        for j, f in enumerate(FEATS):
            a_tr = auc(Z[istr, j], y[istr])
            sgn = 1.0 if a_tr >= 0.5 else -1.0        # 符号は学習期間で決める
            rows.append({"delta_ms": dms, "model": f, "kind": "単変量",
                         "auc_train": a_tr if sgn > 0 else 1 - a_tr,
                         "auc_test": auc(sgn * Z[~istr, j], y[~istr]),
                         "n_test": int((~istr).sum())})
        w = fit_logit(Z[istr], y[istr])
        s_tr = Z[istr] @ w[1:] + w[0]
        s_te = Z[~istr] @ w[1:] + w[0]
        rows.append({"delta_ms": dms, "model": "多変量 (ロジット 10 変数)",
                     "kind": "多変量", "auc_train": auc(s_tr, y[istr]),
                     "auc_test": auc(s_te, y[~istr]),
                     "n_test": int((~istr).sum())})
    A = pl.DataFrame(rows)
    A.write_csv(D / f"latency_auc_{tag}.csv")
    with pl.Config(tbl_rows=80, tbl_width_chars=120):
        print(A.filter(pl.col("kind") == "多変量"))

    # 遅延の実測
    b = pl.read_parquet(D / f"bbo_{tag}.parquet", columns=["ts", "dt"])
    ts = b["ts"].cast(pl.Int64).to_numpy()
    dd = b["dt"].to_numpy()
    u = np.concatenate([[0], np.flatnonzero(np.diff(ts) > 0) + 1])
    g = np.diff(ts[u]) / 1e6
    same = dd[u][1:] == dd[u][:-1]
    g = g[same]
    ps = [0.10, 0.25, 0.50, 0.75, 0.90]
    gp = np.percentile(g, [p * 100 for p in ps])
    wp = wait_pct(g, ps)
    H = pl.read_csv(D / f"hazard_curves_{tag}.csv").filter(
        pl.col("cov") == "all").sort("tbin")
    c0 = float(H["cif_cancel"][0])

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.4),
                             gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})
    fig.suptitle(f"{a.coin}  約定 delta ms 前の標本外 AUC と、実際に取り消せる速さ"
                 f"(学習 {ntr} 日 / 評価 {len(days)-ntr} 日)",
                 color=INK, fontsize=13, y=0.975)
    x = np.array(DELTAS_MS, dtype=float)

    ax = axes[0]
    for f in FEATS:
        v = A.filter((pl.col("model") == f)).sort("delta_ms", descending=True)
        ax.plot(v["delta_ms"].to_numpy(), v["auc_test"].to_numpy(),
                color=FC[f], lw=1.6, marker="o", ms=3.5, label=FN[f])
    v = A.filter(pl.col("kind") == "多変量").sort("delta_ms", descending=True)
    ax.plot(v["delta_ms"].to_numpy(), v["auc_test"].to_numpy(), color=INK,
            lw=2.6, marker="s", ms=5.5, label="多変量 (10 変数)")
    ax.axhline(0.5, color=INK, lw=1.2, ls="--", zorder=1)
    # 横軸は 500ms(左) → 10ms(右) と反転しているので、赤帯は**右側**に出る。
    # 帯の中 (約 200ms 以下) が「取り消しが間に合わない」領域である。
    ax.axvspan(x.min() * 0.8, wp[2], color="#b5322f", alpha=0.10, lw=0,
               zorder=0)
    ax.text(np.sqrt(x.min() * wp[2]), 0.502,
            "この帯の中は取り消しが" + LF + "間に合わない" + LF
            + f"(次の板更新まで中央値 {wp[2]:.0f}ms)", color="#b5322f",
            fontsize=8.5, ha="center", va="bottom")

    style(ax, "約定の何ミリ秒前に判定するか", "標本外 AUC", logx=True)
    ax.set_xticks(DELTAS_MS, [str(v) for v in DELTAS_MS])
    ax.minorticks_off()
    ax.invert_xaxis()
    ax.set_title(f"(a) 標本外 AUC — 赤帯 (≈{wp[2]:.0f}ms 以下) は取り消しが間に合わない",
                 color=INK,
                 fontsize=10, loc="left")
    legend(ax, fs=6.5, loc="upper left", ncol=2)

    ax = axes[1]
    ax.hist(np.clip(g, 0, 800), bins=80, color="#1f5fa8", alpha=0.8, lw=0)
    for p, q, c in zip(ps, gp, ("#c3c2b7", "#9a9890", "#b5322f", "#9a9890",
                                "#c3c2b7")):
        ax.axvline(q, color=c, lw=1.4 if p == 0.5 else 1.0, ls="--")
    ax.text(gp[2] + 12, ax.get_ylim()[1] * 0.86,
            f"中央値 {gp[2]:.0f} ms", color="#b5322f", fontsize=9)
    ax.text(0.42, 0.62, "任意の時点から次のブロックまで\n"
            f"  中央値 {wp[2]:.0f} ms / p90 {wp[4]:.0f} ms",
            color=INK, fontsize=8.5, transform=ax.transAxes)
    style(ax, "ブロック間隔 (ms、800 で打ち切り)", "回数")
    ax.set_title("(b) 板が更新される間隔 = 反応の下限", color=INK, fontsize=10,
                 loc="left")

    ax = axes[2]
    tt = np.array([0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
    cv = []
    for t in tt:
        r = H.filter(pl.col("tau_end_s") <= t)
        cv.append(float(r["cif_cancel"][-1]) if r.height else c0)
    cv = np.array(cv)
    ax.plot(tt * 1000, (cv - c0) / (1 - c0) * 100, color="#b5322f", lw=2.2,
            marker="o", ms=4.5, label="同一ブロックの取消を除く")
    ax.plot(tt * 1000, cv * 100, color="#9a9890", lw=1.4, ls="--",
            label="同一ブロックを含む")
    ax.axhline(c0 * 100, color=MUTED, lw=1.0, ls=":")
    ax.text(11, c0 * 100 + 1.5, f"同一ブロック内 {100*c0:.1f}%", color=MUTED,
            fontsize=8)
    for v_, c in ((50, "#b5322f"), (131, "#1f5fa8")):
        ax.axvline(v_, color=c, lw=1.0, ls="--")
    ax.text(52, 60, "50ms", color="#b5322f", fontsize=8)
    ax.text(135, 60, "ブロック中央値 131ms", color="#1f5fa8", fontsize=8)
    style(ax, "発注からの経過 (ms)", "取り消せた割合 (%)", logx=True)
    ax.set_title("(c) 実際に参加者が取り消せている速さ", color=INK, fontsize=10,
                 loc="left")
    legend(ax, fs=7.5, loc="upper left")

    fig.tight_layout(rect=(0, 0.005, 1, 0.945))
    out = ROOT / "charts" / f"{tag}_latency.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out.name}")
    print(f"\nブロック間隔 p10/25/50/75/90 = "
          f"{'/'.join(f'{v:.0f}' for v in gp)} ms")
    print(f"次のブロックまでの待ち p10/25/50/75/90 = "
          f"{'/'.join(f'{v:.0f}' for v in wp)} ms")


if __name__ == "__main__":
    main()
