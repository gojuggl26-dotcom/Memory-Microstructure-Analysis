"""特徴量の符号の持続性(推移確率行列)を描く。

6 枚。上段が結果、下段が「その結果を信じてよいか」の検査。

  (1) 主役。特徴量 × k の一覧。数字は P(k イベント後も正 | 今が正)、
      色は持続性 = その確率が無条件の確率をどれだけ上回るか
  (2) 減衰曲線・持続する組。帯は日ブロックブートストラップの 95% 区間
  (3) 減衰曲線・符号が反転する組。(2) と縦軸の向きが逆なので分けた
  (4) 3 状態の完全な推移行列。「ちょうど 0」を潰していないことを示す
  (5) 帰無対照の大きさ。0 に潰れ切らない指標があるので、生の超過ではなく
      超過 − 帰無対照 を持続性として読む必要がある
  (6) 恒等式の検算。3 つの特徴量が符号として同一であること、および
      素朴な式が「ちょうど 0」を壊すこと

★タイトルの数値はすべて parquet から計算している。本文からの転記はしない。

配色は scripts/palette_check.py で検証済み(mode=light, surface=#fcfcfb, pairs=all):
  (2) の 4 系列 #d1382f/#12a07a/#5b3fc4/#c98a12
      失敗 0 / 最悪 CVD ΔE 9.5 / 最悪通常視 ΔE 17.7
      #c98a12 のみ対比 2.87 < 3 なので凡例に頼らず直接ラベルを付ける
  (3) の 2 系列 #2a78d6/#b02f7a  失敗 0 / CVD ΔE 15.5 / 通常視 ΔE 25.5
  発散(青↔赤)は plot_obi_ofi.py と同じものを流用(CVD ΔE 21.6 / 通常視 32.3)
  6 系列を 1 枚に混ぜると all-pairs が 2 組落ちるため、(2)(3) に facet した。

    uv run python scripts/plot_sign_persistence.py --coin xyz:MU
出力: charts/<coin>_sign_persistence.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, NEUTRAL = "#e1e0d9", "#f0efec"
DOWN, UP = "#2a78d6", "#e34948"
KS = [1, 2, 3, 4, 5]

# 表示名。OBI / BookSlope / micro_dev は符号が厳密に同一なので 1 行にまとめる
# (同一であることは build_sign_persistence.py が全行で検算している)。
GROUP = {"OBI": "OBI = BookSlope\n= micro_dev", "OFI": "OFI",
         "taker_sign": "テイカーの向き", "ret": "mid のリターン",
         "d_spread": "Δスプレッド幅", "d_depth": "Δ最良気配数量"}
ORDER = ["OBI", "taker_sign", "OFI", "ret", "d_spread", "d_depth"]
KEEP = ["OBI", "taker_sign", "OFI", "ret"]          # (2) 持続する組
FLIP = ["d_spread", "d_depth"]                      # (3) 反転する組
COL = {"OBI": "#d1382f", "taker_sign": "#12a07a", "OFI": "#5b3fc4",
       "ret": "#c98a12", "d_spread": "#2a78d6", "d_depth": "#b02f7a"}
MK = {"OBI": "o", "taker_sign": "s", "OFI": "^", "ret": "D",
      "d_spread": "o", "d_depth": "s"}
SHORT = {"OBI": "OBI 群", "taker_sign": "テイカー", "OFI": "OFI",
         "ret": "リターン", "d_spread": "Δスプレッド", "d_depth": "Δ数量"}
STATES = ["負", "ちょうど 0", "正"]


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, lw=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def get(C, feat, k=None, col=None):
    s = C.filter((pl.col("feat") == feat) & (pl.col("day_type") == "全日"))
    if k is not None:
        s = s.filter(pl.col("k") == k)
    return s[col][0] if col else s.sort("k")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")

    C = pl.read_parquet(ROOT / "data" / f"sign_persist_cells_{tag}.parquet")
    I = pl.read_parquet(ROOT / "data" / f"sign_persist_ident_{tag}.parquet")
    n_days = get(C, "OBI", 1, "n_days")
    n_pairs = get(C, "OBI", 1, "n_pairs")

    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False, "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })
    fig, axes = plt.subplots(2, 3, figsize=(17.6, 10.2))

    # ---- (1) 主役: 特徴量 × k --------------------------------------------
    ax = axes[0, 0]
    P = np.array([[get(C, f, k, "p_pos_given_pos") * 100 for k in KS] for f in ORDER])
    E = np.array([[get(C, f, k, "excess_adj") * 100 for k in KS] for f in ORDER])
    cmap = LinearSegmentedColormap.from_list("d", [DOWN, NEUTRAL, UP])
    norm = TwoSlopeNorm(vcenter=0, vmin=min(E.min(), -1), vmax=E.max())
    ax.imshow(E, cmap=cmap, norm=norm, aspect="auto")
    for i in range(len(ORDER)):
        for j in range(len(KS)):
            ax.text(j, i - 0.13, f"{P[i, j]:.1f}%", ha="center", va="center",
                    fontsize=10.5, color=INK, weight="bold")
            # 濃く塗ったセルの上では INK2 が沈むので、地の濃さで色を切り替える。
            # 発散配色は正負で伸びが違う(vmax +25.6 / vmin -7.6)ので、
            # 片側ずつ正規化しないと最も濃い青のセルを取りこぼす。
            deep = (E[i, j] / E.max() if E[i, j] > 0 else E[i, j] / E.min()) > 0.62
            ax.text(j, i + 0.22, f"{E[i, j]:+.1f}pp", ha="center", va="center",
                    fontsize=8, color=SURFACE if deep else INK2)
    ax.set_xticks(range(len(KS)))
    ax.set_xticklabels([f"k={k}" for k in KS], fontsize=9.5)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([GROUP[f] for f in ORDER], fontsize=8.5)
    ax.set_xlabel("何イベント後を見るか", color=INK2, fontsize=10)
    ax.grid(False)
    top = ORDER[int(np.argmax(E[:, 0]))]
    ax.set_title(
        f"(1) 太字 = P(k 後も正 | 今が正)、小字 = 持続性\n"
        f"最大は OBI 群の k=1 で {P[ORDER.index(top), 0]:.1f}%"
        f"(持続性 {E[ORDER.index(top), 0]:+.1f}pp)",
        loc="left", color=INK, fontsize=10.5, pad=10, weight="bold")

    # ---- (2) 減衰: 持続する組 --------------------------------------------
    ax = axes[0, 1]
    style(ax)
    for f in KEEP:
        s = get(C, f)
        y = s["excess_adj"].to_numpy() * 100
        lo = s["excess_adj_lo"].to_numpy() * 100
        hi = s["excess_adj_hi"].to_numpy() * 100
        ax.fill_between(KS, lo, hi, color=COL[f], alpha=0.16, lw=0)
        ax.plot(KS, y, MK[f] + "-", color=COL[f], lw=2, ms=7,
                label=GROUP[f].replace("\n", " "))
        # OFI と mid のリターンは k=5 で 0.5pp 差しかなく、直接ラベルが重なる。
        # 縦にずらして読めるようにする。
        dy = {"OFI": 7, "ret": -7}.get(f, 0)
        ax.annotate(SHORT[f], (KS[-1], y[-1]),
                    xytext=(7, dy), textcoords="offset points", va="center",
                    fontsize=8.5, color=COL[f], weight="bold")
    ax.axhline(0, color=MUTED, lw=1.2, ls="--")
    ax.set_xticks(KS); ax.set_xlim(0.85, 6.1)
    ax.set_xlabel("何イベント後 k", color=INK2, fontsize=10)
    ax.set_ylabel("持続性[pp](帯は 95% 区間)", color=INK2, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    r_obi = get(C, "OBI")["excess_adj"].to_numpy() * 100
    half = r_obi[0] / 2
    ax.set_title(
        f"(2) 減るが 5 イベント先でも 0 に届かない\n"
        f"OBI 群 {r_obi[0]:+.1f} → {r_obi[-1]:+.1f}pp"
        f"(半減 k≈{np.interp(half, r_obi[::-1], np.array(KS, float)[::-1]):.1f})",
        loc="left", color=INK, fontsize=10.5, pad=10, weight="bold")

    # ---- (3) 減衰: 反転する組 --------------------------------------------
    ax = axes[0, 2]
    style(ax)
    for f in FLIP:
        s = get(C, f)
        y = s["excess_adj"].to_numpy() * 100
        lo = s["excess_adj_lo"].to_numpy() * 100
        hi = s["excess_adj_hi"].to_numpy() * 100
        ax.fill_between(KS, lo, hi, color=COL[f], alpha=0.18, lw=0)
        ax.plot(KS, y, MK[f] + "-", color=COL[f], lw=2, ms=7,
                label=GROUP[f])
        ax.annotate(SHORT[f], (KS[-1], y[-1]), xytext=(7, 0),
                    textcoords="offset points", va="center", fontsize=8.5,
                    color=COL[f], weight="bold")
    ax.axhline(0, color=MUTED, lw=1.4, ls="--")
    ax.set_xticks(KS); ax.set_xlim(0.85, 6.1)
    ax.set_xlabel("何イベント後 k", color=INK2, fontsize=10)
    ax.set_ylabel("持続性[pp](帯は 95% 区間)", color=INK2, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    sp = get(C, "d_spread")["excess_adj"].to_numpy() * 100
    dp = get(C, "d_depth")["excess_adj"].to_numpy() * 100
    # 符号が変わる k を数値で拾う(本文から転記しない)
    flip_k = next((k for k, v in zip(KS, sp) if v > 0), None)
    ax.set_title(
        f"(3) この 2 つは k=1 で負 — 直後は戻る\n"
        f"Δスプレッドは {sp[0]:+.1f}pp、k={flip_k} で正へ"
        f"(Δ数量は k=5 で {dp[-1]:+.2f}pp)",
        loc="left", color=INK, fontsize=10.5, pad=10, weight="bold")

    # ---- (4) 3 状態の完全な推移行列 --------------------------------------
    ax = axes[1, 0]
    M = np.array([[get(C, "OBI", 1, f"p_{i}{j}") * 100 for j in range(3)]
                  for i in range(3)])
    N = [get(C, "OBI", 1, f"n_from_{i}") for i in range(3)]
    seq = LinearSegmentedColormap.from_list("s", [SURFACE, UP])
    ax.imshow(M, cmap=seq, vmin=0, vmax=100, aspect="auto")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{M[i, j]:.1f}%", ha="center", va="center",
                    fontsize=12, weight="bold",
                    color=SURFACE if M[i, j] > 55 else INK)
    ax.set_xticks(range(3)); ax.set_xticklabels(STATES, fontsize=9.5)
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"{s}\n({n:,})" for s, n in zip(STATES, N)], fontsize=8.5)
    ax.set_xlabel("1 イベント後の符号", color=INK2, fontsize=10)
    ax.set_ylabel("今の符号(件数)", color=INK2, fontsize=10)
    ax.grid(False)
    z_share = N[1] / sum(N) * 100
    ax.set_title(
        f"(4) OBI の 3 状態推移行列(k=1)。行の合計 100%\n"
        f"「ちょうど 0」は {z_share:.2f}%、次も 0 が {M[1, 1]:.1f}%"
        f" — 2 状態に潰すと消える",
        loc="left", color=INK, fontsize=10.5, pad=10, weight="bold")

    # ---- (5) 帰無対照の大きさ --------------------------------------------
    ax = axes[1, 1]
    style(ax)
    raw = np.array([get(C, f, 1, "excess") * 100 for f in ORDER])
    nul = np.array([get(C, f, 1, "excess_perm") * 100 for f in ORDER])
    x = np.arange(len(ORDER))
    ax.bar(x - 0.19, raw, width=0.36, color=[COL[f] for f in ORDER])
    ax.bar(x + 0.19, nul, width=0.36, color=MUTED)
    ax.axhline(0, color=INK2, lw=1.1)
    for xi, v in zip(x, nul):
        ax.annotate(f"{v:+.2f}", (xi + 0.19, v), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=8,
                    color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[f] for f in ORDER], fontsize=8.5,
                       rotation=20, ha="right")
    ax.set_ylabel("k=1 の超過[pp]", color=INK2, fontsize=10)
    # 左の棒の色は特徴量の識別に使っているので、色見本の凡例だと食い違う。
    # 2 つの系列は位置で決まるため、最初の対に直接ラベルを置く。
    ax.annotate("生の超過", (x[0] - 0.19, raw[0]), xytext=(-2, 8),
                textcoords="offset points", ha="center", fontsize=8.5,
                color=INK2, weight="bold")
    ax.annotate("帰無対照", (x[4] + 0.19, nul[4]), xytext=(-6, 62),
                textcoords="offset points", ha="center", fontsize=8.5,
                color=MUTED, weight="bold",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9,
                                shrinkB=2))
    worst = ORDER[int(np.argmax(np.abs(nul)))]
    ax.set_title(
        f"(5) 帰無対照は 0 に潰れ切らない(全日合算の偏り)\n"
        f"最大 {SHORT[worst]} の {nul[np.argmax(np.abs(nul))]:+.2f}pp"
        f" — 持続性は生の超過からこれを引く",
        loc="left", color=INK, fontsize=10.5, pad=10, weight="bold")

    # ---- (6) 恒等式の検算 ------------------------------------------------
    ax = axes[1, 2]
    style(ax)
    agg = (I.group_by("pair")
           .agg(pl.col("n_mismatch").sum(), pl.col("n").sum(),
                pl.col("n_mismatch_at_zero").sum())
           .sort("n_mismatch"))
    lab = {"OBI vs BookSlope": "BookSlope\n(q^b−q^a)/h",
           "OBI vs micro_dev": "micro_dev\n因数分解した式",
           "OBI vs _micro_naive": "micro_dev\n素朴な引き算"}
    ys = np.arange(agg.height)
    vals = agg["n_mismatch"].to_numpy().astype(float)
    ax.barh(ys, vals, color=[MUTED if v == 0 else UP for v in vals], height=0.5)
    tot = int(agg["n"][0])
    for y, v in zip(ys, vals):
        ax.annotate("0 件 — 符号が完全一致" if v == 0
                    else f"{int(v):,} 件({v / tot * 100:.2f}%)",
                    (v, y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9,
                    color=INK2 if v == 0 else UP, weight="bold")
    ax.set_yticks(ys)
    ax.set_yticklabels([lab[p] for p in agg["pair"]], fontsize=8.5)
    ax.set_xlim(0, max(vals) * 1.75 if max(vals) else 1)
    ax.set_xlabel(f"OBI と符号が食い違った行数(全 {tot:,} 行)",
                  color=INK2, fontsize=10)
    bad = agg.filter(pl.col("pair") == "OBI vs _micro_naive")
    nb, zb = int(bad["n_mismatch"][0]), int(bad["n_mismatch_at_zero"][0])
    ax.set_title(
        f"(6) 3 つは符号として同一 — 式の書き方で壊れる\n"
        f"素朴な引き算の {nb:,} 件は {zb / nb * 100:.0f}% が"
        f"「ちょうど 0」の行",
        loc="left", color=INK, fontsize=10.5, pad=10, weight="bold")

    fig.suptitle(
        f"{a.coin} 特徴量の符号は何イベント先まで持続するか — "
        f"{n_days} 日 / BBO {n_pairs:,} 組(k=1)",
        x=0.008, ha="left", color=INK, fontsize=13.5, weight="bold")
    fig.text(0.008, 0.013,
             "各セルは P(sign x_{t+k} = 正 | sign x_t = 正)。持続性 = その確率 − 同じ組上の無条件 P(正) − 帰無対照。"
             "これは x 自身の自己相関であって将来価格の予測力ではない。"
             "テイカーの向きだけイベントの時計が違う(成行注文単位)ので k を横に比べないこと。",
             color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=[0, 0.030, 1, 0.945])
    out = ROOT / "charts" / f"{tag}_sign_persistence.png"
    fig.savefig(out, dpi=170)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
