"""複数の口座が同時に動くか — 8 指標を組み立てて描く。

    uv run python scripts/plot_sync.py --coin xyz:MU
出力: charts/<coin>_sync_block.png   ブロック粒度の同時性(指標 1・2)
      charts/<coin>_sync_corr.png    秒粒度の相関(指標 3・4・5)
      charts/<coin>_sync_common.png  共通成分と herding score(指標 6・7・8)
      data/sync_pairs_<coin>.csv     口座ペアごとの共起比と相関
      data/sync_wallet_<coin>.csv    口座ごとの共通成分と herding score

【読み方】
すべて**帰無との比**で見る。この市場ではブロック粒度の同時性も、秒粒度の
相関も、構造的に高く出る。

    ブロック … 観測共起 ÷ 分ごとに層化した独立の期待値。1 なら「偶然どおり」
    秒     … 観測相関 と ±300 秒ずらしの帰無相関 を並べる。差が同期の分
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
OBS, NUL, WARN = "#e34948", "#2a78d6", "#eb6834"
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
SEC = 86_400
MIN_D = 10                 # ペアの統計を出すのに要る「両者が活動した日数」
SLAB = {"add": "発注本数", "can": "取消本数", "flow": "符号つき出し入れ(契約)",
        "dist": "発注価格の mid からの距離(bp)"}


def _pow10(v, _p=None):
    if v <= 0:
        return ""
    e = int(round(math.log10(v)))
    if abs(v - 10 ** e) > 1e-9 * max(v, 1):
        return ""
    if e == 0:
        return "1"
    if 1 <= e <= 4:
        return f"{10 ** e:,}"
    return ("0.1" if e == -1 else "0.01" if e == -2
            else "10" + str(e).translate(SUP))


def style(ax, xlab="", ylab="", logx=False, logy=False):
    for lg, axis, setter in ((logx, ax.xaxis, ax.set_xscale),
                             (logy, ax.yaxis, ax.set_yscale)):
        if lg:
            setter("log")
            axis.set_major_locator(LogLocator(base=10.0))
            axis.set_major_formatter(FuncFormatter(_pow10))
            axis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", color=GRID, lw=0.6, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(BASELINE)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.set_xlabel(xlab, color=INK2, fontsize=9)
    ax.set_ylabel(ylab, color=INK2, fontsize=9)


def legend(ax, **kw):
    lg = ax.legend(fontsize=8, frameon=True, facecolor=SURFACE, edgecolor=GRID,
                   labelcolor=INK2, **kw)
    lg.get_frame().set_linewidth(0.6)
    return lg


def pair_corr(sxy, sx, sxx, days):
    """指定した日だけを足して Pearson を出す。days は bool のマスク。"""
    T = SEC * days.sum()
    if T < SEC * MIN_D:
        return np.nan
    Sxy = sxy[days].sum()
    return Sxy, T


def build(tag: str):
    z = np.load(ROOT / "data" / f"sync_raw_{tag}.npz", allow_pickle=True)
    wid = z["wid"]
    n = len(wid)
    nd = len(z["dt"])
    act = {k: z[f"act_{k}"] for k in ("add", "can")}       # (nd, n)
    alive = act["add"] > 0                                  # (nd, n) 活動した日

    # ---- 1・2 ブロック粒度の共起比 ---------------------------------------
    lift = {}
    for nm in ("add", "can"):
        O, E = z[f"obs_{nm}"], z[f"exp_{nm}"]
        L = np.full((n, n), np.nan)
        nds = np.zeros((n, n), int)
        for i in range(n):
            for j in range(i + 1, n):
                d = alive[:, i] & alive[:, j]
                nds[i, j] = nds[j, i] = d.sum()
                if d.sum() < MIN_D:
                    continue
                e = E[d, i, j].sum()
                if e > 0:
                    L[i, j] = L[j, i] = O[d, i, j].sum() / e
        lift[nm] = L
        lift[f"n_{nm}"] = nds

    # ---- 3・4・5 秒粒度の相関(観測と帰無)-------------------------------
    corr, corr0 = {}, {}
    for s in ("add", "can", "flow", "dist"):
        SXY, SX, SXX, NUL_ = (z[f"sxy_{s}"], z[f"sx_{s}"], z[f"sxx_{s}"],
                              z[f"null_{s}"])
        R = np.full((n, n), np.nan)
        R0 = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(i + 1, n):
                d = alive[:, i] & alive[:, j]
                if d.sum() < MIN_D:
                    continue
                T = SEC * d.sum()
                sx, sy = SX[d, i].sum(), SX[d, j].sum()
                vx = SXX[d, i].sum() - sx * sx / T
                vy = SXX[d, j].sum() - sy * sy / T
                if vx <= 0 or vy <= 0:
                    continue
                den = math.sqrt(vx * vy)
                R[i, j] = R[j, i] = (SXY[d, i, j].sum() - sx * sy / T) / den
                R0[i, j] = R0[j, i] = (NUL_[d, i, j].sum() - sx * sy / T) / den
        corr[s], corr0[s] = R, R0

    # ---- 5' 市場共通の反応を除いた残差の相関 -----------------------------
    #   自分たち 2 者を**除いた**市場合計 M を統制する。M に自分を入れると
    #   機械的に負へ引っ張られるので必ず除く。
    #     r_ij|M = (r_ij − r_iM r_jM) / sqrt((1−r_iM²)(1−r_jM²))
    pcorr, pcorr0 = {}, {}
    for s in ("add", "can", "flow", "dist"):
        SXY, SX, SXX, NUL_ = (z[f"sxy_{s}"], z[f"sx_{s}"], z[f"sxx_{s}"],
                              z[f"null_{s}"])
        for tag_, M_ in (("obs", SXY), ("nul", NUL_)):
            R = np.full((n, n), np.nan)
            for i in range(n):
                for j in range(i + 1, n):
                    d = alive[:, i] & alive[:, j]
                    if d.sum() < MIN_D:
                        continue
                    T = SEC * d.sum()
                    sx = SX[d].sum(axis=0)
                    C = M_[d].sum(axis=0) - np.outer(sx, sx) / T
                    # 自分の分散はずらしても変わらないので常に実測を使う
                    for k in (i, j):
                        C[k, k] = SXX[d, k].sum() - sx[k] ** 2 / T
                    Tt = C.sum()
                    Ri, Rj = C[i].sum(), C[j].sum()
                    vm = Tt - 2 * Ri - 2 * Rj + C[i, i] + 2 * C[i, j] + C[j, j]
                    if vm <= 0 or C[i, i] <= 0 or C[j, j] <= 0:
                        continue
                    cim = Ri - C[i, i] - C[i, j]
                    cjm = Rj - C[j, j] - C[i, j]
                    rim = cim / math.sqrt(C[i, i] * vm)
                    rjm = cjm / math.sqrt(C[j, j] * vm)
                    rij = C[i, j] / math.sqrt(C[i, i] * C[j, j])
                    den = math.sqrt(max(1 - rim * rim, 1e-12)
                                    * max(1 - rjm * rjm, 1e-12))
                    R[i, j] = R[j, i] = (rij - rim * rjm) / den
            (pcorr if tag_ == "obs" else pcorr0)[s] = R

    # ---- 6・7 共通成分(自分を除いた市場合計への決定係数)------------------
    cm, cm0 = {}, {}
    for s in ("add", "can", "flow", "dist"):
        SXY, SX, SXX, NUL_ = (z[f"sxy_{s}"], z[f"sx_{s}"], z[f"sxx_{s}"],
                              z[f"null_{s}"])
        r2 = np.full(n, np.nan)
        r20 = np.full(n, np.nan)
        for i in range(n):
            d = alive[:, i]
            if d.sum() < MIN_D:
                continue
            T = SEC * d.sum()
            sx = SX[d].sum(axis=0)                        # (n,)
            for tag_, M in (("obs", SXY), ("nul", NUL_)):
                C = M[d].sum(axis=0) - np.outer(sx, sx) / T
                # 自分の分散だけは常に実測(ずらしても不変)
                C[i, i] = SXX[d, i].sum() - sx[i] ** 2 / T
                if C[i, i] <= 0:
                    continue
                o = np.ones(n, bool)
                o[i] = False
                cov = C[i, o].sum()
                var = C[np.ix_(o, o)].sum()
                if var <= 0:
                    continue
                v = cov * cov / (C[i, i] * var)
                if tag_ == "obs":
                    r2[i] = v
                else:
                    r20[i] = v
        cm[s], cm0[s] = r2, r20
    return (z, wid, n, nd, alive, act, lift, corr, corr0, cm, cm0,
            pcorr, pcorr0)


def rankpct(v):
    out = np.full(len(v), np.nan)
    m = np.isfinite(v)
    if m.sum() > 1:
        out[m] = np.argsort(np.argsort(v[m])) / (m.sum() - 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    (z, wid, n, nd, alive, act, lift, corr, corr0, cm, cm0,
     pcorr, pcorr0) = build(tag)
    ordn = act["add"].sum(axis=0)                  # 活動ブロック数(規模の代理)
    iu = np.triu_indices(n, 1)

    # ---- 表 ---------------------------------------------------------------
    rows = []
    for i, j in zip(*iu):
        if lift["n_add"][i, j] < MIN_D:
            continue
        rows.append({"wid_i": int(wid[i]), "wid_j": int(wid[j]),
                     "n_days": int(lift["n_add"][i, j]),
                     "lift_add": lift["add"][i, j], "lift_can": lift["can"][i, j],
                     **{f"corr_{s}": corr[s][i, j] for s in corr},
                     **{f"null_{s}": corr0[s][i, j] for s in corr0},
                     **{f"pcorr_{s}": pcorr[s][i, j] for s in pcorr},
                     **{f"pnull_{s}": pcorr0[s][i, j] for s in pcorr0}})
    P = pl.DataFrame(rows)
    P.write_csv(ROOT / "data" / f"sync_pairs_{tag}.csv")

    herd_src = np.column_stack([cm[s] - cm0[s] for s in ("add", "can", "flow",
                                                         "dist")])
    herd = np.nanmean(np.column_stack([rankpct(herd_src[:, k])
                                       for k in range(4)]), axis=1)
    Wd = pl.DataFrame({
        "wid": wid, "active_days": alive.sum(axis=0), "n_blocks_add": ordn,
        **{f"cm_{s}": cm[s] for s in cm},
        **{f"cm_null_{s}": cm0[s] for s in cm0},
        **{f"mean_corr_{s}": np.nanmean(np.where(np.eye(n, dtype=bool), np.nan,
                                                 corr[s]), axis=1)
           for s in corr},
        "herding_score": herd})
    Wd.write_csv(ROOT / "data" / f"sync_wallet_{tag}.csv")

    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False, "text.parse_math": False,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "legend.framealpha": 0.92})
    out = ROOT / "charts"
    ok = np.isfinite(lift["add"]) & (lift["n_add"] >= MIN_D)
    iu2 = iu

    # ================= 図 1: ブロック粒度 ==================================
    fig = plt.figure(figsize=(16.4, 9.4), dpi=155)
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.28,
                          left=0.055, right=0.975, top=0.845, bottom=0.085)
    for k, nm in enumerate(("add", "can")):
        ax = fig.add_subplot(gs[0, k])
        v = lift[nm][iu2]                       # ★上三角のみ(二重計上を防ぐ)
        v = v[np.isfinite(v) & (v > 0)]
        ax.hist(v, bins=np.logspace(-1, 1.5, 60), color=OBS, zorder=3)
        ax.axvline(1, color=INK, lw=1.4, zorder=4)
        ax.set_title(f"① {'発注' if nm == 'add' else '取消'}の同時性 — "
                     f"共起 ÷ 期待値", color=INK, fontsize=10.5, pad=7, loc="left")
        style(ax, "比(1 = 偶然どおり)", "ペア数" if k == 0 else "", logx=True)
        ax.text(0.03, 0.95,
                f"ペア {len(v):,} / 中央値 {np.median(v):.2f}\n"
                f"1 を超えるペア {100 * (v > 1).mean():.0f}%",
                transform=ax.transAxes, va="top", fontsize=8.6, color=INK2)

    ax = fig.add_subplot(gs[0, 2])
    x = lift["add"][ok]
    y = lift["can"][ok]
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    ax.plot(x[m], y[m], "o", ms=3.2, color=MUTED, alpha=0.55, zorder=3)
    lim = [max(np.nanpercentile(x[m], 0.5), 0.05), np.nanpercentile(x[m], 99.5)]
    ax.plot(lim, lim, "-", color=INK, lw=1.0, zorder=4)
    r = np.corrcoef(np.log(x[m]), np.log(y[m]))[0, 1]
    ax.set_title(f"② 発注と取消で同じペアが高いか(log 相関 {r:+.2f})",
                 color=INK, fontsize=10.5, pad=7, loc="left")
    style(ax, "発注の比", "取消の比", logx=True, logy=True)

    for k, nm in enumerate(("add", "can")):
        ax = fig.add_subplot(gs[1, k])
        L = np.where(ok, lift[nm], np.nan)
        v = np.log2(np.clip(L, 1e-2, 1e2))
        im = ax.imshow(v, cmap="RdBu_r", vmin=-2, vmax=2)
        ax.set_title(f"③ {'発注' if nm == 'add' else '取消'}の比(log₂、"
                     f"口座は注文数の多い順)", color=INK, fontsize=10.5, pad=7,
                     loc="left")
        ax.tick_params(colors=INK2, labelsize=7)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label("log₂(共起 ÷ 期待値)", color=INK2, fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    small = np.minimum.outer(ordn, ordn)[ok]
    m = np.isfinite(x) & (x > 0) & (small > 0)
    ax.plot(small[m], x[m], "o", ms=3.2, color=MUTED, alpha=0.55, zorder=3)
    ax.axhline(1, color=INK, lw=1.2, zorder=4)
    ax.set_title("④ 比は活動量の産物か(横 = 少ないほうの活動ブロック数)",
                 color=INK, fontsize=10.5, pad=7, loc="left")
    style(ax, "活動ブロック数(対数)", "発注の比(対数)", logx=True, logy=True)

    fig.text(0.055, 0.972, "ブロック粒度の同時性 — 同じブロックに一緒に動くか",
             fontsize=17, color=INK, va="top")
    fig.text(0.055, 0.930,
             f"上位 {n} 口座。期待値は**分ごとに層化**した独立の場合の共起数で、"
             "日内の活動プロファイルは吸収されている。1 なら偶然どおり。",
             fontsize=10, color=INK2, va="top")
    p = out / f"{tag}_sync_block.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {p}", file=sys.stderr)

    # ================= 図 2: 秒粒度の相関 ==================================
    fig = plt.figure(figsize=(16.4, 9.4), dpi=155)
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.26,
                          left=0.055, right=0.975, top=0.845, bottom=0.085)
    for k, s in enumerate(("add", "can", "flow", "dist")):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        v = corr[s][iu]
        v0 = corr0[s][iu]
        m = np.isfinite(v) & np.isfinite(v0)
        bins = np.linspace(-0.4, 1.0, 80)
        ax.hist(v0[m], bins=bins, color=NUL, alpha=0.65, label="帰無(±300秒ずらし)",
                zorder=3)
        ax.hist(v[m], bins=bins, color=OBS, alpha=0.65, label="観測", zorder=4)
        ax.axvline(0, color=INK, lw=1.0, zorder=5)
        ax.set_title(f"{'①②③④'[k]} {SLAB[s]}", color=INK, fontsize=10.5,
                     pad=7, loc="left")
        style(ax, "ペアの相関", "ペア数" if k % 2 == 0 else "")
        legend(ax, loc="upper right")
        ax.text(0.03, 0.95,
                f"観測 中央値 {np.median(v[m]):+.3f}\n帰無 中央値 {np.median(v0[m]):+.3f}",
                transform=ax.transAxes, va="top", fontsize=8.6, color=INK2)

    ax = fig.add_subplot(gs[0, 2])
    C = np.where(np.eye(n, dtype=bool), np.nan, corr["add"])
    im = ax.imshow(C, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax.set_title("⑤ 発注本数の相関行列", color=INK, fontsize=10.5, pad=7,
                 loc="left")
    ax.tick_params(colors=INK2, labelsize=7)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

    ax = fig.add_subplot(gs[1, 2])
    y_ = np.arange(4)
    ss = ("add", "can", "flow", "dist")
    exc = [np.nanmedian((corr[s] - corr0[s])[iu]) for s in ss]
    pex = [np.nanmedian((pcorr[s] - pcorr0[s])[iu]) for s in ss]
    ax.barh(y_ - 0.2, exc, height=0.38, color=OBS, label="観測 − 帰無", zorder=3)
    ax.barh(y_ + 0.2, pex, height=0.38, color=INK,
            label="市場共通を除いた残差(観測 − 帰無)", zorder=3)
    ax.axvline(0, color=INK, lw=1.0, zorder=4)
    ax.set_yticks(y_)
    ax.set_yticklabels([SLAB[s] for s in ss], fontsize=8)
    ax.yaxis.tick_right()                      # 左隣の図と重ならないように
    ax.invert_yaxis()
    ax.set_title("⑥ 同期の分 — 市場共通を除く前と後", color=INK, fontsize=10.5,
                 pad=7, loc="left")
    style(ax, "相関の差", "")
    legend(ax, loc="lower right")

    fig.text(0.055, 0.972, "秒粒度の相関 — 一緒に出し、一緒に引くか",
             fontsize=17, color=INK, va="top")
    fig.text(0.055, 0.930,
             "1 秒ごとの系列をペアで相関。帰無は各口座の系列を ±300 秒ずらしたもの"
             "(日内の山は保ち、秒の対応だけ壊す)。差が同期の分。",
             fontsize=10, color=INK2, va="top")
    p = out / f"{tag}_sync_corr.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {p}", file=sys.stderr)

    # ================= 図 3: 共通成分と herding ============================
    fig = plt.figure(figsize=(16.4, 9.4), dpi=155)
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.26,
                          left=0.055, right=0.975, top=0.845, bottom=0.085)
    for k, s in enumerate(("add", "can")):
        ax = fig.add_subplot(gs[0, k])
        o = np.sort(cm[s][np.isfinite(cm[s])])
        u = np.sort(cm0[s][np.isfinite(cm0[s])])
        ax.plot(o, np.arange(len(o)) / max(len(o), 1) * 100, color=OBS, lw=2.0,
                label="観測", zorder=3)
        ax.plot(u, np.arange(len(u)) / max(len(u), 1) * 100, color=NUL, lw=2.0,
                label="帰無", zorder=3)
        nm = "replenishment(発注)" if s == "add" else "cancellation(取消)"
        ax.set_title(f"① common-mode {nm}", color=INK, fontsize=10.5, pad=7,
                     loc="left")
        style(ax, "自分を除く市場合計への決定係数", "累積%(口座)" if k == 0 else "")
        ax.set_ylim(0, 100)
        legend(ax, loc="lower right")

    ax = fig.add_subplot(gs[0, 2])
    for s, cc in (("add", OBS), ("can", NUL)):
        C = corr[s].copy()
        m = np.isfinite(C).all(axis=1) | True
        idx = np.where(np.isfinite(np.nanmean(np.where(np.eye(n, dtype=bool),
                                                       np.nan, C), axis=1)))[0]
        if len(idx) < 3:
            continue
        Cs = np.nan_to_num(C[np.ix_(idx, idx)], nan=0.0)
        np.fill_diagonal(Cs, 1.0)
        ev = np.sort(np.linalg.eigvalsh(Cs))[::-1]
        ax.plot(np.arange(1, len(ev) + 1), ev / ev.sum() * 100, "o-", color=cc,
                ms=4, lw=1.7, label=SLAB[s], zorder=3)
    ax.set_title("② 相関行列の固有値(第 1 主成分が共通成分)", color=INK,
                 fontsize=10.5, pad=7, loc="left")
    style(ax, "順位", "分散の説明率(%)")
    legend(ax, loc="upper right")

    ax = fig.add_subplot(gs[1, 0])
    v = np.sort(herd[np.isfinite(herd)])
    ax.plot(v, np.arange(len(v)) / max(len(v), 1) * 100, color=INK, lw=2.0,
            zorder=3)
    ax.set_title("③ maker herding score の分布", color=INK, fontsize=10.5,
                 pad=7, loc="left")
    style(ax, "score(相対順位の平均)", "累積%(口座)")
    ax.set_ylim(0, 100)

    ax = fig.add_subplot(gs[1, 1])
    m = np.isfinite(herd) & (ordn > 0)
    ax.plot(ordn[m], herd[m], "o", ms=5, color=MUTED, alpha=0.75, zorder=3)
    r = (np.corrcoef(np.log10(ordn[m]), herd[m])[0, 1] if m.sum() > 3 else np.nan)
    ax.set_title(f"④ score は規模の言い換えか(相関 {r:+.3f})", color=INK,
                 fontsize=10.5, pad=7, loc="left")
    style(ax, "活動ブロック数(対数)", "herding score", logx=True)

    ax = fig.add_subplot(gs[1, 2])
    y_ = np.arange(4)
    ss = ("add", "can", "flow", "dist")
    ob = [np.nanmedian(cm[s]) for s in ss]
    nu = [np.nanmedian(cm0[s]) for s in ss]
    ax.barh(y_ - 0.2, ob, height=0.38, color=OBS, label="観測", zorder=3)
    ax.barh(y_ + 0.2, nu, height=0.38, color=NUL, label="帰無", zorder=3)
    ax.set_yticks(y_)
    ax.set_yticklabels([SLAB[s] for s in ss], fontsize=8)
    ax.yaxis.tick_right()
    ax.invert_yaxis()
    ax.set_title("⑤ 共通成分の決定係数(中央値)", color=INK, fontsize=10.5,
                 pad=7, loc="left")
    style(ax, "決定係数", "")
    legend(ax, loc="lower right")

    fig.text(0.055, 0.972, "共通成分と herding — 市場全体と一緒に動く分",
             fontsize=17, color=INK, va="top")
    fig.text(0.055, 0.930,
             "決定係数は**自分を除いた**市場合計への回帰。自分を含めると"
             "大きい口座ほど機械的に上がる。帰無は ±300 秒ずらし。",
             fontsize=10, color=INK2, va="top")
    p = out / f"{tag}_sync_common.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {p}", file=sys.stderr)
    print(f"-> data/sync_pairs_{tag}.csv({P.height} ペア) / "
          f"sync_wallet_{tag}.csv({Wd.height} 口座)", file=sys.stderr)


if __name__ == "__main__":
    main()
