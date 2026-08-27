"""攻撃的な売買(テイカー)の向きが、次の向きにどうつながるかを推移確率行列にする。

## 何を 1 つの「イベント」と数えるか

素朴に約定 1 件を 1 イベントとすると、**1 本の成行注文が板を何段も食っただけ**で
同じ向きが連続してしまい、持続性を過大評価する。実測では 11,174,755 件の
テイカー約定が 6,372,171 本の成行注文にまとまり、1 本あたり平均 1.75 件だった。

そこで 2 通りを併記する。

| 単位 | 数え方 | イベント数 |
|---|---|---|
| 約定単位 | テイカー約定 1 件 = 1 イベント | 11,174,755 |
| **成行注文単位** | 同一 ns・同一ユーザーの約定をまとめて 1 イベント | 6,372,171 |

**成行注文単位のほうが「意思決定の連なり」に近い**ので、こちらを主として読む。

## 推移確率行列

直前 k 個の向きを状態とし、次の向きが買いになる確率を出す。

    P(次 = 買い | 直前 k 個の並び)     状態は 2^k 個

k を 1 つ増やしたときに情報が増えるかを尤度比検定(G 検定)で判定する。
自由度は 2^(k-1)。検定を K 回行うので Bonferroni で閾値を割る。

## ★ 標本が大きすぎることへの注意

イベントが 637 万件あるので、**ごく僅かな偏りでも有意になる**。有意性だけを
見て「効果がある」と判断してはいけないので、各次数で
**効果量(親の状態からの確率のずれの最大値)**を併記する。

## 帰無対照

向きの並びを無作為に入れ替えた系列でも同じ計算をする。持続性が本物なら
入れ替え後には消えるはずである。

## 日境界

日をまたいで連鎖をつながない(前日の最後と当日の最初は別の連なりとして扱う)。

x が確定する時刻 / y の期間: 説明変数は直前 k 個の向きで、目的変数は次の 1 個。
説明変数は目的変数より前に確定しているので先読みは無い。

    uv run python scripts/build_sign_chain.py --coin xyz:MU
出力: data/sign_chain_<coin>.csv, charts/<coin>_sign_chain_*.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
BUY_C, SELL_C, NULL_C = "#2a78d6", "#eb6834", "#898781"
KMAX = 14


def sequences(coin: str) -> dict:
    """日ごとの向き系列を、約定単位と成行注文単位で作る。買い = 1、売り = 0。"""
    tag = coin.replace(":", "_")
    f = (
        pl.read_parquet(ROOT / "data" / f"fills_{tag}.parquet",
                        columns=["ts", "user", "side", "sz", "crossed"])
        .filter(pl.col("crossed"))
        .sort("ts")
        .with_columns(d=pl.col("ts").dt.date(), b=(pl.col("side") == "B").cast(pl.Int8))
    )
    # 1 本の成行が板を複数段食った場合、その約定群は同じ向きのはず。検算する。
    chk = f.group_by("ts", "user").agg(u=pl.col("b").n_unique())
    mixed = int((chk["u"] > 1).sum())
    print(f"[検算] 同一 ns・同一ユーザーで向きが混在するグループ: {mixed:,} / {chk.height:,}")

    order = (
        f.group_by("ts", "user")
        .agg(b=pl.col("b").first(), sz=pl.col("sz").sum(), d=pl.col("d").first())
        # ★同一 ns に別々のイベントが並ぶ(全体の 29.4%)。group_by は順序を保証
        # しないので、user で決定的に並べ替えないと実行ごとに結果が変わる。
        # 同一 ns 内の前後関係に経済的な意味は無いため、順序は任意で構わないが
        # 再現できることは必要。並べ方への感度は tie_sensitivity() で測る。
        .sort("ts", "user")
    )
    out = {}
    for name, df in (("約定単位", f), ("成行注文単位", order)):
        seqs = [g["b"].to_numpy().astype(np.int8)
                for _, g in df.sort("ts", "user").group_by("d", maintain_order=True)]
        out[name] = seqs
        n = sum(len(s) for s in seqs)
        share = sum(int(s.sum()) for s in seqs) / n
        print(f"[{name}] {n:,} イベント / {len(seqs)} 日 / 買いの割合 {share * 100:.2f}%")
    return out


def tie_sensitivity(coin: str, reps: int = 10) -> None:
    """同一 ns 内の並べ方を無作為に変えたとき、推移確率がどれだけ動くかを測る。"""
    tag = coin.replace(":", "_")
    f = (
        pl.read_parquet(ROOT / "data" / f"fills_{tag}.parquet",
                        columns=["ts", "user", "side", "crossed"])
        .filter(pl.col("crossed"))
        .with_columns(d=pl.col("ts").dt.date(), b=(pl.col("side") == "B").cast(pl.Int8))
        .group_by("ts", "user").agg(b=pl.col("b").first(), d=pl.col("d").first())
    )
    rng = np.random.default_rng(12345)
    vals = {"買→買": [], "売→売": [], "買買→買": [], "売売→売": []}
    for _ in range(reps):
        g = f.with_columns(r=pl.Series(rng.random(f.height))).sort("ts", "r")
        seqs = [x["b"].to_numpy().astype(np.int8)
                for _, x in g.group_by("d", maintain_order=True)]
        cs = counts_all_orders(seqs, 2)
        p1 = cs[0][:, 1] / cs[0].sum(axis=1)
        p2 = cs[1][:, 1] / cs[1].sum(axis=1)
        vals["買→買"].append(p1[1] * 100)
        vals["売→売"].append((1 - p1[0]) * 100)
        vals["買買→買"].append(p2[3] * 100)
        vals["売売→売"].append((1 - p2[0]) * 100)
    print()
    print(f"=== 同一 ns の並べ方への感度({reps} 通りの無作為な並べ方)===")
    for k, v in vals.items():
        v = np.array(v)
        print(f"  {k}: {v.min():.2f}% 〜 {v.max():.2f}%(幅 {v.max() - v.min():.2f} ポイント)")


def counts_all_orders(seqs: list, kmax: int) -> list:
    """k = 1..kmax の [売りが来た数, 買いが来た数] を一度の走査でまとめて返す。

    状態は直前 k 個の向きを 2 進で符号化したもの(古い方が上位ビット)。
    位置 i を予測するときの状態は s[i-k .. i-1] で、

        state_(k+1)(i) = state_k(i) + (s[i-k-1] << k)

    と 1 つ古い向きを最上位に足すだけで伸びる。k ごとに数え直さずこの差分で
    進める(入れ替え検定で何本も回すため速さが要る)。
    """
    out = [np.zeros((2 ** k, 2), dtype=np.int64) for k in range(1, kmax + 1)]
    for s0 in seqs:
        s0 = s0.astype(np.int64)
        n = len(s0)
        if n < 2:
            continue
        st = s0[:-1]                       # k = 1 の状態。i = 1..n-1 に対応
        for k in range(1, kmax + 1):
            if len(st) == 0:
                break
            np.add.at(out[k - 1], (st, s0[k:]), 1)   # st は i = k..n-1 に対応
            if k < kmax:
                st = st[1:] + (s0[: len(st) - 1] << k)
    return out


def counts_at_order(seqs: list, k: int) -> np.ndarray:
    """1 つの次数だけ数える(表示用)。"""
    return counts_all_orders(seqs, k)[k - 1]


def g_test(child: np.ndarray, parent: np.ndarray) -> tuple:
    """次数 k の当てはめが k-1 より良いかの尤度比検定。child は 2^k x 2。"""
    p_par = parent / np.maximum(parent.sum(axis=1, keepdims=True), 1)
    # child の状態 i(k ビット)の親は下位 k-1 ビット
    par_of = np.arange(child.shape[0]) & (parent.shape[0] - 1)
    exp = child.sum(axis=1, keepdims=True) * p_par[par_of]
    m = (child > 0) & (exp > 0)
    g = 2.0 * float((child[m] * np.log(child[m] / exp[m])).sum())
    df = parent.shape[0]                        # 2^(k-1)
    p_child = child / np.maximum(child.sum(axis=1, keepdims=True), 1)
    n_ok = child.sum(axis=1) >= 100             # 標本の薄い状態は効果量から除く
    eff = float(np.abs(p_child[n_ok, 1] - p_par[par_of][n_ok, 1]).max()) if n_ok.any() else 0.0
    return g, df, eff


def chi2_sf(x: float, df: int) -> float:
    """カイ二乗分布の上側確率。"""
    from scipy import stats
    return float(stats.chi2.sf(x, df)) if x > 0 else 1.0


def run_lengths(seqs: list, nmax: int = 12) -> np.ndarray:
    """同じ向きが n 回続いた直後に、また同じ向きが来る割合。"""
    cont = np.zeros(nmax + 1, dtype=np.int64)
    tot = np.zeros(nmax + 1, dtype=np.int64)
    for s in seqs:
        if len(s) < 2:
            continue
        run = 1
        for i in range(1, len(s)):
            n = min(run, nmax)
            tot[n] += 1
            if s[i] == s[i - 1]:
                cont[n] += 1
                run += 1
            else:
                run = 1
    return np.vstack([tot, cont])


def main() -> None:
    import math
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    seqs = sequences(a.coin)

    # --- 次数の検定 ---------------------------------------------------------
    # 標本が 637 万件あるので、カイ二乗の漸近 p 値は何でも 0 になり役に立たない。
    # しかも向きを入れ替えた系列でも漸近検定は棄却してしまう(日ごとに買いの
    # 割合が違うため、日をまたいで集めると見かけの依存が出る)。
    # そこで **入れ替え検定** に置き換える。日の中だけで向きを無作為に並べ替えた
    # 系列を R 本作り、その G 統計量の分布を帰無分布として実データと比べる。
    rng = np.random.default_rng(0)
    R = 20
    rows = []
    for name in ("約定単位", "成行注文単位"):
        base = seqs[name]
        cs = counts_all_orders(base, KMAX)
        real, prev = [], None
        for k in range(1, KMAX + 1):
            c = cs[k - 1]
            parent = np.array([c.sum(axis=0)]) if prev is None else prev
            real.append(g_test(c, parent))
            prev = c

        null = np.zeros((R, KMAX))
        for r in range(R):
            cs_r = counts_all_orders([rng.permutation(x) for x in base], KMAX)
            prev = None
            for k in range(1, KMAX + 1):
                c = cs_r[k - 1]
                parent = np.array([c.sum(axis=0)]) if prev is None else prev
                null[r, k - 1] = g_test(c, parent)[0]
                prev = c

        print()
        print(f"=== {name}: 次数 k を 1 つ増やすと情報が増えるか(入れ替え検定 R = {R})===")
        print(f"{'k':>2} {'実データ G':>13} {'帰無の平均':>11} {'帰無の 95%点':>13} "
              f"{'倍率':>7} {'効果量':>8}  判定")
        last_sig = 0
        for k in range(1, KMAX + 1):
            g, df, eff = real[k - 1]
            nm, n95 = float(null[:, k - 1].mean()), float(np.quantile(null[:, k - 1], 0.95))
            ok = g > n95
            if ok:
                last_sig = k
            print(f"{k:>2} {g:13,.0f} {nm:11,.0f} {n95:13,.0f} {g / max(nm, 1e-9):7.1f} "
                  f"{eff:8.4f}  {'有意' if ok else '有意でない'}")
            rows.append({"unit": name, "k": k, "G": g, "df": df, "effect": eff,
                         "null_mean": nm, "null_p95": n95, "significant": ok})
        print(f"  → 帰無分布を超える最大の次数: k = {last_sig}")

    pl.DataFrame(rows).write_csv(ROOT / "data" / f"sign_chain_{tag}.csv")

    # --- 1 次と 2 次の推移確率行列を表示 ------------------------------------
    for name in ("約定単位", "成行注文単位"):
        for k in (1, 2, 3):
            c = counts_at_order(seqs[name], k)
            p = c[:, 1] / c.sum(axis=1)
            print(f"\n=== {name} の {k} 次推移確率(次が買いになる確率)===")
            for i in range(2 ** k):
                st = "".join("買" if (i >> (k - 1 - j)) & 1 else "売" for j in range(k))
                n = int(c[i].sum())
                se = math.sqrt(p[i] * (1 - p[i]) / n)
                print(f"  直前 {st} → 買い {p[i] * 100:6.2f}%  売り {100 - p[i] * 100:6.2f}%"
                      f"  (n = {n:>10,}, 標準誤差 {se * 100:.3f}%)")

    # --- 連続回数ごとの継続確率 ----------------------------------------------
    rl = {name: run_lengths(seqs[name]) for name in ("約定単位", "成行注文単位")}
    print("\n=== 同じ向きが n 回続いた直後に、また同じ向きが来る割合 ===")
    print(f"{'n':>3} {'約定単位':>22} {'成行注文単位':>22}")
    for n in range(1, 11):
        cells = []
        for name in ("約定単位", "成行注文単位"):
            t, c = rl[name][0][n], rl[name][1][n]
            cells.append(f"{c / t * 100:6.2f}% (n={t:>9,})" if t else "—")
        print(f"{n:>3} {cells[0]:>22} {cells[1]:>22}")

    # --- 作図 ---------------------------------------------------------------
    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False, "text.parse_math": False,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })

    fig, ax = plt.subplots(figsize=(11.5, 6.0), dpi=170)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=1)
    ax.set_axisbelow(True)
    xs = np.arange(1, 11)
    for name, col in (("約定単位", SELL_C), ("成行注文単位", BUY_C)):
        t, c = rl[name][0][1:11].astype(float), rl[name][1][1:11].astype(float)
        pr = c / t
        se = np.sqrt(pr * (1 - pr) / t)
        ax.fill_between(xs, (pr - 1.96 * se) * 100, (pr + 1.96 * se) * 100,
                        color=col, alpha=0.18, lw=0, zorder=2)
        ax.plot(xs, pr * 100, color=col, lw=2.2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3, label=name)
    ax.axhline(50, color=NULL_C, lw=1.4, zorder=2)
    ax.text(1.0, 50.4, "向きに記憶が無い場合の 50%", color=MUTED, fontsize=9,
            va="bottom", ha="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)
    ax.set_xticks(xs)
    ax.set_xlabel("直前に同じ向きが続いた回数 n", color=INK2, fontsize=10.5)
    ax.set_ylabel("次も同じ向きになる割合(%)", color=INK2, fontsize=10.5)
    ax.legend(loc="lower right", frameon=False, fontsize=10, labelcolor=INK2)
    ax.set_title(f"{a.coin} 攻撃的な売買の向きの continuation 確率",
                 loc="left", color=INK, fontsize=13, pad=30, weight="bold")
    ax.text(0, 1.045,
            "帯は 95% 信頼区間。成行注文単位は、同一 ns・同一ユーザーの約定を 1 本の注文として"
            "まとめたもの(板を複数段食っただけの連続を除くため)。",
            transform=ax.transAxes, color=INK2, fontsize=9.5)
    fig.text(0.005, 0.012, "出所: Hyperliquid L4 (Artemis) node_fills / 窓 2026-05-04〜08-10",
             color=MUTED, fontsize=8)
    fig.subplots_adjust(left=0.08, right=0.975, top=0.855, bottom=0.115)
    out = ROOT / "charts" / f"{tag}_sign_chain_continuation.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"\n[chart] {out}")

    # 3 次の推移確率行列(成行注文単位)をヒートマップに
    k = 3
    c = counts_at_order(seqs["成行注文単位"], k)
    p = c[:, 1] / c.sum(axis=1)
    fig, ax = plt.subplots(figsize=(9.0, 6.4), dpi=170)
    m = np.vstack([1 - p, p]).T
    # 発散配色は検証済みパレットの赤 ↔ 青、中間は無彩色。50% が「どちらでもない」と
    # 読めるようにするため、中間に色相を置く配色(RdYlBu 等)は使わない。
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "buy_sell", ["#e34948", "#f0efec", "#2a78d6"])
    im = ax.imshow(m, cmap=cmap, vmin=0.30, vmax=0.70, aspect="auto")
    labels = ["".join("買" if (i >> (k - 1 - j)) & 1 else "売" for j in range(k))
              for i in range(2 ** k)]
    ax.set_yticks(range(2 ** k), labels, fontsize=11)
    ax.set_xticks([0, 1], ["次が売り", "次が買い"], fontsize=11)
    for i in range(2 ** k):
        for j in range(2):
            ax.text(j, i, f"{m[i, j] * 100:.2f}%", ha="center", va="center",
                    color=INK, fontsize=11, weight="bold")
    ax.tick_params(colors=INK2, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_ylabel("直前 3 本の向き(古い順)", color=INK2, fontsize=10.5)
    ax.set_title(f"{a.coin} 3 次の推移確率行列(成行注文単位)",
                 loc="left", color=INK, fontsize=13, pad=30, weight="bold")
    share = float(c[:, 1].sum() / c.sum()) * 100
    ax.text(0, 1.045,
            f"全体では買いが {share:.2f}%。"
            "値そのものは各セルに書いてある。色は 30〜70% を振り切りとし、中間の 50% は無彩色。",
            transform=ax.transAxes, color=INK2, fontsize=9.5)
    fig.colorbar(im, ax=ax, shrink=0.7, label="確率")
    fig.subplots_adjust(left=0.16, right=0.99, top=0.855, bottom=0.07)
    out2 = ROOT / "charts" / f"{tag}_sign_chain_matrix.png"
    fig.savefig(out2)
    print(f"[chart] {out2}")

    tie_sensitivity(a.coin)


if __name__ == "__main__":
    main()
