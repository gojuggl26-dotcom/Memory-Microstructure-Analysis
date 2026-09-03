"""注文の生存率・取消ハザード・約定ハザードを 6 つの条件別に描く。

    uv run python scripts/plot_hazard.py --coin xyz:MU
出力: charts/<coin>_hazard_survival.png   生存率 S(τ)
      charts/<coin>_hazard_cancel.png     取消ハザード(毎秒)
      charts/<coin>_hazard_fill.png       約定ハザード(毎秒)
      charts/<coin>_hazard_controls.png   競合リスク・帰無対照・距離を揃えた対照
      data/hazard_curves_<coin>.csv       上の 4 図の数値
      data/hazard_summary_<coin>.csv      層ごとの要約(前半・後半つき)

【推定】打ち切りを扱える生命表(actuarial)方式。ビン j について

    n_j  = 開始時点で生きている本数
    n'_j = n_j − 0.5 × (そのビン内の打ち切り数)
    q_j  = (約定 + 取消)_j / n'_j              … 全原因の消滅確率
    S_j  = Π_{k<j} (1 − q_k)                   … 生存率
    h_f  = 約定_j / (n'_j × Δt_j)              … 原因別ハザード(毎秒)
    h_c  = 取消_j / (n'_j × Δt_j)
    F_f(j+1) = F_f(j) + S_j × 約定_j / n'_j    … 累積約定確率

取消と約定は競合リスクなので、「取消率」を単独で 1 − S と読んではいけない。
累積発生確率 F は両方の原因を同時に扱った上での「最終的にそうなる割合」である。

【配色】順序のある層(queue / size / dist / vol / wallet)は**単一色相の連続階調**
(薄い = 小さい、濃い = 大きい)。符号のある層(ofi)は**発散**(青 = 逆向き、
灰 = 無流量、赤 = 順向き)で、色覚特性下でも読めるよう逆向きを破線にする。
階調は `palette_check.py` で背景との対比 3.0 以上を確認済み。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_hazard import COVS, TAU_EDGES, NT, NEAR_TICK  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
FILLC, CANCELC = "#e34948", "#2a78d6"      # 発散ペア: 赤 = 約定 / 青 = 取消
HB, HR = 4.547787, 0.502944                # OKLCH 色相(#2a78d6 / #e34948)
MIN_RISK = 500      # これ未満の危険集合になったビンは描かない
MIN_EVT = 5         # その原因の件数がこれ未満のビンも描かない(対数軸で暴れるため)

TITLE = {"queue": "① 自分より前に並んでいた数量(契約)",
         "size": "② 自分の注文数量(契約)",
         "dist": "③ 同じ側の最良気配からの距離(ティック)",
         "wallet": "④ その口座のそれまでの累計注文本数",
         "vol": "⑤ 直前 300 秒の実現ボラティリティ",
         "ofi": "⑥ 直前 1 秒の OFI(自分の注文へ向かう向きを正)"}
SHORT = {"queue": "前に並んでいた数量", "size": "注文数量", "dist": "距離",
         "wallet": "口座の累計本数", "vol": "ボラティリティ", "ofi": "OFI"}
TAG = {"queue": "前", "size": "量", "dist": "距", "wallet": "口座",
       "vol": "ボラ", "ofi": "OFI"}
PANELS = ["queue", "size", "dist", "wallet", "vol", "ofi"]
SUBNOTE = ("板に留まる指値(tif が Alo・Gtc、トリガー注文と reduce_only を除く)。"
           "層の値はすべて発注の瞬間までに確定している量なので、先読みは無い。")


# ---- 連続階調と発散階調(OKLab で作る)---------------------------------
def _ok2hex(L: float, a: float, b: float) -> str:
    def f(c):
        return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    rgb = (4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
           -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
           -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s)
    return "#" + "".join(f"{max(0, min(255, round(f(max(x, 0)) * 255))):02x}" for x in rgb)


def seq(n, h=HB, L0=0.66, L1=0.28, C0=0.10, C1=0.16):
    out = []
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0.0
        L, C = L0 + (L1 - L0) * t, C0 + (C1 - C0) * t
        out.append(_ok2hex(L, C * math.cos(h), C * math.sin(h)))
    return out


def div(n):
    k = (n - 1) // 2
    return seq(k, HB, 0.32, 0.62, 0.16, 0.10) + [MUTED] + seq(n - 1 - k, HR, 0.62, 0.38, 0.10, 0.19)


def colors_for(cov: str, n: int):
    if cov == "ofi":
        return div(n), [(0, (5, 2))] * ((n - 1) // 2) + ["-"] * (n - (n - 1) // 2)
    if cov == "null":
        return [MUTED] * n, ["-"] * n
    return seq(n), ["-"] * n


# ---- 生命表 -------------------------------------------------------------
def life_table(c: np.ndarray) -> dict:
    """c[tbin, 0:3] = (約定, 取消, 打切) → 生存率・原因別ハザード・累積発生確率。"""
    df, dc, dz = c[:, 0].astype(float), c[:, 1].astype(float), c[:, 2].astype(float)
    N = df.sum() + dc.sum() + dz.sum()
    left = np.concatenate([[0.0], np.cumsum(df + dc + dz)[:-1]])
    n = N - left
    ne = np.maximum(n - 0.5 * dz, 1e-9)
    q = np.clip((df + dc) / ne, 0, 1)
    S = np.concatenate([[1.0], np.cumprod(1 - q)])          # S[j] = ビン j の開始時点
    dt = np.diff(np.concatenate([TAU_EDGES, [TAU_EDGES[-1] * 10]])) / 1e9   # 秒
    with np.errstate(divide="ignore", invalid="ignore"):
        hf = np.where(dt > 0, df / (ne * dt), np.nan)
        hc = np.where(dt > 0, dc / (ne * dt), np.nan)
    cf = np.cumsum(S[:-1] * df / ne)
    cc = np.cumsum(S[:-1] * dc / ne)
    return {"N": N, "n": n, "ne": ne, "S": S[1:], "hf": hf, "hc": hc,
            "cif_f": cf, "cif_c": cc, "df": df, "dc": dc, "dz": dz}


def median_life(lt) -> float:
    """生存率が 0.5 を切る時刻。ビン内は log 時間で線形に按分する。"""
    S = lt["S"]
    j = int(np.searchsorted(-S, -0.5))
    if j >= len(S):
        return float("nan")
    s0 = 1.0 if j == 0 else S[j - 1]
    t0 = TAU_EDGES[j] / 1e9
    t1 = TAU_END[j]
    if s0 <= 0.5 or t0 <= 0:
        return float(t1)
    w = (s0 - 0.5) / max(s0 - S[j], 1e-12)
    return float(np.exp(np.log(t0) + w * np.log(t1 / t0)))


def cells_to_arr(C: pl.DataFrame, cov: str, nb: int) -> np.ndarray:
    a = np.zeros((nb, NT, 3), dtype=np.int64)
    s = C.filter(pl.col("cov") == cov)
    if s.height:
        a[s["bin"].to_numpy(), s["tbin"].to_numpy(), 0] = s["d_fill"].to_numpy()
        a[s["bin"].to_numpy(), s["tbin"].to_numpy(), 1] = s["d_cancel"].to_numpy()
        a[s["bin"].to_numpy(), s["tbin"].to_numpy(), 2] = s["d_cens"].to_numpy()
    return a


TAU_MID = np.sqrt(np.maximum(TAU_EDGES, 1.0)
                  * np.concatenate([TAU_EDGES[1:], [TAU_EDGES[-1] * 10]])) / 1e9
TAU_END = np.concatenate([TAU_EDGES[1:], [TAU_EDGES[-1] * 10]]) / 1e9
XLIM = (0.03, 3e4)
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _pow10(v, _pos=None):
    """mathtext を切っているので指数は unicode の上付きで書く。"""
    if v <= 0:
        return ""
    e = int(round(math.log10(v)))
    if abs(v - 10 ** e) > 1e-9 * max(v, 1):
        return ""
    if e == 0:
        return "1"
    if 1 <= e <= 4:
        return f"{10 ** e:,}"
    if e == -1:
        return "0.1"
    if e == -2:
        return "0.01"
    return "10" + str(e).translate(SUP)


def style(ax, xlab="注文が置かれてからの経過時間(秒、対数目盛)", ylab="", logy=False):
    ax.set_xscale("log")
    ax.set_xlim(*XLIM)
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(FuncFormatter(_pow10))
    ax.xaxis.set_minor_formatter(NullFormatter())
    if logy:
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(FuncFormatter(_pow10))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", color=GRID, lw=0.6, zorder=0)
    ax.grid(True, which="minor", color=GRID, lw=0.3, alpha=0.5, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(BASELINE)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.set_xlabel(xlab, color=INK2, fontsize=9)
    ax.set_ylabel(ylab, color=INK2, fontsize=9)


def draw_family(ax, arr, labs, cov, key, logy=False):
    cols, dashes = colors_for(cov, len(labs))
    x = TAU_END if key == "S" else TAU_MID
    for i, lab in enumerate(labs):
        lt = life_table(arr[i])
        if lt["N"] < 5000:
            continue
        y = lt[key]
        m = (lt["ne"] >= MIN_RISK) & np.isfinite(y) & (x >= XLIM[0]) & (x <= XLIM[1])
        if key in ("hf", "hc"):
            m &= lt["df" if key == "hf" else "dc"] >= MIN_EVT
        if m.sum() < 2:
            continue
        ax.plot(x[m], y[m], color=cols[i], lw=1.9, ls=dashes[i],
                label=f"{lab}  n={lt['N']/1e6:.2f}M" if lt["N"] >= 1e6
                else f"{lab}  n={lt['N']/1e3:.0f}k", zorder=3, solid_capstyle="round")
    if logy:
        ax.set_yscale("log")
    if not ax.get_lines():
        ax.text(0.5, 0.5, "データなし", transform=ax.transAxes, ha="center",
                va="center", fontsize=11, color=MUTED)
        return
    leg = ax.legend(fontsize=7.4, loc="upper right" if key == "S" else "best",
                    frameon=True, facecolor=SURFACE,
                    edgecolor=GRID, labelcolor=INK2, handlelength=1.6,
                    borderpad=0.4, labelspacing=0.28)
    leg.get_frame().set_linewidth(0.6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    C = pl.read_parquet(ROOT / "data" / f"hazard_cells_{tag}.parquet")
    arrs = {k: cells_to_arr(C, k, len(v)) for k, v in COVS.items()}

    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False, "text.parse_math": False,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "legend.framealpha": 0.92,
    })
    outdir = ROOT / "charts"
    outdir.mkdir(exist_ok=True)

    # ---- 図 1〜3: 6 条件 × (生存率 / 取消ハザード / 約定ハザード) ---------
    specs = [
        ("survival", "S", False, "板に残っている割合",
         "指値注文の生存率 — 発注時に判る 6 つの条件で層別",
         "縦軸は「まだ板に残っている割合」。横軸は対数。曲線が早く落ちるほど短命。"),
        ("cancel", "hc", True, "取消ハザード(毎秒)",
         "取消ハザード — そこまで生き残った注文が次の瞬間に取り消される率",
         "縦軸は毎秒の率(対数)。1.0 なら「1 秒あたり 1 回」の勢い。"
         "件数が 5 未満のビンは描いていない。"),
        ("fill", "hf", True, "約定ハザード(毎秒)",
         "約定ハザード — そこまで生き残った注文が次の瞬間に約定する率",
         "縦軸は毎秒の率(対数)。取消と約定は競合リスクとして同時に扱っている。"
         "件数が 5 未満のビンは描いていない。"),
    ]
    for name, key, logy, ylab, title, sub in specs:
        fig = plt.figure(figsize=(16.4, 10.2), dpi=155)
        gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.20,
                              left=0.055, right=0.985, top=0.855, bottom=0.075)
        for k, cov in enumerate(PANELS):
            ax = fig.add_subplot(gs[k // 3, k % 3])
            draw_family(ax, arrs[cov], COVS[cov], cov, key, logy)
            ax.set_title(TITLE[cov], color=INK, fontsize=10.5, pad=7, loc="left")
            style(ax, ylab=ylab if k % 3 == 0 else "", logy=logy)
            if key == "S":
                ax.set_ylim(0, 1.02)
                if k == 0:
                    ax.annotate("左端が 1.0 に届かないのは、発注と同じブロックで\n"
                                "消えた注文(全体の約 9%)がここに含まれるため",
                                xy=(0.022, 0.90), xytext=(0.055, 0.30),
                                textcoords="axes fraction", fontsize=7.6, color=MUTED,
                                arrowprops=dict(arrowstyle="-", color=BASELINE, lw=0.8))
        fig.text(0.055, 0.972, title, fontsize=17, color=INK, va="top")
        fig.text(0.055, 0.936, sub, fontsize=10, color=INK2, va="top")
        fig.text(0.055, 0.912, SUBNOTE, fontsize=8.8, color=MUTED, va="top")
        p = outdir / f"{tag}_hazard_{name}.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        print(f"-> {p}", file=sys.stderr)

    # ---- 図 4: 競合リスクと対照 -------------------------------------------
    # 右列は 33 行の層名を縦に並べるので、3 行ぶんの高さを 1 枚で使う。
    fig = plt.figure(figsize=(16.4, 14.4), dpi=150)
    gs = fig.add_gridspec(3, 3, hspace=0.34, wspace=0.30,
                          left=0.055, right=0.985, top=0.895, bottom=0.050,
                          width_ratios=[1.0, 1.0, 1.05])

    ax = fig.add_subplot(gs[0, 0])
    lt = life_table(arrs["all"][0])
    m = (lt["ne"] >= MIN_RISK) & (TAU_END >= XLIM[0]) & (TAU_END <= XLIM[1])
    ax.plot(TAU_END[m], lt["S"][m], color=INK, lw=2.0, label="板に残っている")
    ax.plot(TAU_END[m], lt["cif_c"][m], color=CANCELC, lw=2.0, label="取り消された(累積)")
    ax.plot(TAU_END[m], lt["cif_f"][m], color=FILLC, lw=2.0, label="約定した(累積)")
    band = 1.96 / np.sqrt(np.maximum(np.cumsum(lt["df"]), 1))
    ax.fill_between(TAU_END[m], lt["cif_f"][m] * (1 - band[m]),
                    lt["cif_f"][m] * (1 + band[m]), color=FILLC, alpha=0.18, lw=0)
    ax.text(2.4e4, lt["cif_c"][m][-1] - 0.06, f"取消 {lt['cif_c'][m][-1]:.1%}",
            ha="right", va="top", fontsize=9, color=CANCELC)
    ax.text(2.4e4, lt["cif_f"][m][-1] + 0.05, f"約定 {lt['cif_f'][m][-1]:.2%}",
            ha="right", va="bottom", fontsize=9, color=FILLC)
    ax.set_title("全体 — 3 つの行き先の内訳(競合リスク)", color=INK, fontsize=10.5,
                 pad=7, loc="left")
    style(ax, ylab="割合")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, loc="center left", frameon=True, facecolor=SURFACE,
              edgecolor=GRID, labelcolor=INK2)

    ax = fig.add_subplot(gs[0, 1])
    draw_family(ax, arrs["null"], COVS["null"], "null", "S")
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_title("帰無対照 — 無作為に 8 層へ振ったとき", color=INK, fontsize=10.5,
                 pad=7, loc="left")
    style(ax)
    ax.set_ylim(0, 1.0)
    ax.text(0.97, 0.93, "8 本が重なる = 層別の仕組み自体は\n差を作っていない",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK2)

    ax = fig.add_subplot(gs[:, 2])
    rows, ys = [], []
    for cov in ("queue", "size", "wallet", "vol", "ofi"):
        near = f"near_{cov}"
        for i, lab in enumerate(COVS[cov]):
            a1 = arrs[cov][i]
            a2 = arrs[near][i]
            n1, n2 = a1.sum(), a2.sum()
            if n1 < 5000:
                continue
            rows.append((f"{TAG[cov]} {lab}", a1[:, 0].sum() / max(n1, 1),
                         a2[:, 0].sum() / max(n2, 1) if n2 >= 5000 else np.nan))
    y = np.arange(len(rows))
    a_all = np.array([r[1] for r in rows])
    a_near = np.array([r[2] for r in rows])
    ok = np.isfinite(a_near)
    ax.hlines(y[ok], a_all[ok], a_near[ok], color=BASELINE, lw=1.4, zorder=2)
    ax.plot(a_all, y, "o", ms=5.5, color=BASELINE, mec=SURFACE, mew=1.2,
            ls="none", label="すべての注文", zorder=3)
    ax.plot(a_near[ok], y[ok], "o", ms=5.5, color=FILLC, mec=SURFACE, mew=1.2,
            ls="none", label=f"最良から {NEAR_TICK:.0f} ティック以内だけ", zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.0)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(FuncFormatter(_pow10))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("最終的に約定した割合", color=INK2, fontsize=9)
    ax.grid(True, axis="x", color=GRID, lw=0.6, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_title("距離を揃えると効果はどれだけ残るか", color=INK, fontsize=10.5,
                 pad=7, loc="left")
    ax.legend(fontsize=7.4, loc="lower left", frameon=True, facecolor=SURFACE,
              edgecolor=GRID, labelcolor=INK2, numpoints=1)

    for k, cov in enumerate(("queue", "size", "vol", "ofi")):
        ax = fig.add_subplot(gs[1 + k // 2, k % 2])
        draw_family(ax, arrs[f"near_{cov}"], COVS[cov], cov, "hf", True)
        ax.set_title(f"{SHORT[cov]} — 最良 {NEAR_TICK:.0f} ティック以内に限った"
                     f"約定ハザード", color=INK, fontsize=10.5, pad=7, loc="left")
        style(ax, ylab="約定ハザード(毎秒)" if k % 2 == 0 else "", logy=True)

    fig.text(0.055, 0.980, "検算 — 競合リスクの内訳・帰無対照・距離を揃えた対照",
             fontsize=17, color=INK, va="top")
    fig.text(0.055, 0.951,
             "左上は「最後にどうなったか」の内訳。中上は層別そのものが差を作って"
             "いないことの確認。右上と下段は、距離が他の 5 条件すべてと相関するため"
             "距離を揃えた比較。", fontsize=10, color=INK2, va="top")
    p = outdir / f"{tag}_hazard_controls.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {p}", file=sys.stderr)

    # ---- 数値の書き出し ---------------------------------------------------
    out = []
    for cov, labs in COVS.items():
        for i, lab in enumerate(labs):
            lt = life_table(arrs[cov][i])
            if lt["N"] < 1:
                continue
            for j in range(NT):
                if lt["n"][j] <= 0:
                    continue
                out.append({"cov": cov, "bin_lab": lab, "tbin": j,
                            "tau_end_s": float(TAU_END[j]), "n_risk": float(lt["n"][j]),
                            "d_fill": int(lt["df"][j]), "d_cancel": int(lt["dc"][j]),
                            "d_cens": int(lt["dz"][j]), "S": float(lt["S"][j]),
                            "h_fill": float(lt["hf"][j]), "h_cancel": float(lt["hc"][j]),
                            "cif_fill": float(lt["cif_f"][j]),
                            "cif_cancel": float(lt["cif_c"][j])})
    pl.DataFrame(out).write_csv(ROOT / "data" / f"hazard_curves_{tag}.csv")
    print(f"-> data/hazard_curves_{tag}.csv", file=sys.stderr)

    # ---- 層ごとの要約。前半・後半に割った値も並べる(プールした順序が
    #      どこか 1 期間の異常に支配されていないかを見るため)------------
    hp = ROOT / "data" / f"hazard_cells_half_{tag}.parquet"
    H = pl.read_parquet(hp) if hp.exists() else None
    rows = []
    for cov, labs in COVS.items():
        for i, lab in enumerate(labs):
            lt = life_table(arrs[cov][i])
            if lt["N"] < 1000:
                continue
            r = {"cov": cov, "bin_lab": lab, "n": int(lt["N"]),
                 "p_tau0": float((lt["df"][0] + lt["dc"][0]) / max(lt["N"], 1)),
                 "fill_final": float(lt["cif_f"][-1]),
                 "cancel_final": float(lt["cif_c"][-1]),
                 "median_life_s": median_life(lt),
                 "S_1s": float(np.interp(1.0, TAU_END, lt["S"])),
                 "S_60s": float(np.interp(60.0, TAU_END, lt["S"])),
                 "h_fill_1s": float(np.interp(1.0, TAU_MID, np.nan_to_num(lt["hf"])))}
            if H is not None:
                for k in (0, 1):
                    a = cells_to_arr(H.filter(pl.col("half") == k), cov, len(labs))
                    l2 = life_table(a[i])
                    r[f"fill_h{k}"] = float(l2["cif_f"][-1]) if l2["N"] else float("nan")
                    r[f"median_h{k}"] = median_life(l2) if l2["N"] else float("nan")
            rows.append(r)
    pl.DataFrame(rows).write_csv(ROOT / "data" / f"hazard_summary_{tag}.csv")
    print(f"-> data/hazard_summary_{tag}.csv", file=sys.stderr)


if __name__ == "__main__":
    main()
