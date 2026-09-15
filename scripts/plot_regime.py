"""6 レジーム判定(図)。

    uv run python scripts/plot_regime.py

入力: data/regime_share.csv / regime_fwd.csv / regime_summary.csv
      (`analyze_regime.py`)
出力: charts/allcoins_regime.png

配色: 銘柄を 11 色に塗り分けない(CVD 分離が取れない)。
**実測は C1 の細線の束、帰無対照(ブートストラップ)は CM の点線**で重ねる。
主張は「実測と対照がどれだけ離れているか」なので、この 2 色だけで足りる。
費用のパネルだけ C2(粗利)と INK(往復費用)を使う。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _chartstyle import C1, C2, C3, CM, INK, D, plt, save  # noqa: E402

SHORT = ["R1\n低/中", "R2\n低/上", "R3\n低/下",
         "R4\n高/上", "R5\n高/下", "R6\n高/両"]


def lines(ax, S, col, coins, **kw):
    """銘柄ごとの折れ線(6 レジーム)。"""
    x = np.arange(6)
    for c in coins:
        q = S.filter(pl.col("coin") == c).sort("regime")
        ax.plot(x, q[col].to_numpy(), **kw)
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT)


def main() -> None:
    S = pl.read_csv(D / "regime_share.csv")
    F = pl.read_csv(D / "regime_fwd.csv")
    M = pl.read_csv(D / "regime_summary.csv")
    coins = M["coin"].to_list()
    real = dict(color=C1, lw=1.0, alpha=0.65)
    null = dict(color=CM, lw=1.0, alpha=0.8, ls=":")

    fig, ax = plt.subplots(2, 3, figsize=(16.6, 9.0))

    # A. 割合
    b = ax[0, 0]
    lines(b, S, "share", coins, **real)
    lines(b, S, "share_null", coins, **null)
    b.set_title("A. レジームの割合", loc="left")
    b.set_ylabel("割合")
    b.plot([], [], color=C1, lw=1.6, label=f"実測({len(coins)} 銘柄)")
    b.plot([], [], color=CM, lw=1.6, ls=":", label="帰無対照(ブートストラップ)")
    b.legend(frameon=False, fontsize=8)

    # B. 平均継続秒
    b = ax[0, 1]
    lines(b, S, "run_s", coins, **real)
    lines(b, S, "run_s_null", coins, **null)
    b.set_title("B. 平均継続秒 — 実測のほうが長い = 状態である", loc="left")
    b.set_ylabel("秒")

    # C. 自己遷移確率
    b = ax[0, 2]
    lines(b, S, "diag", coins, **real)
    lines(b, S, "diag_null", coins, **null)
    b.set_title("C. 次の 1 秒も同じレジームでいる確率", loc="left")
    b.set_ylabel("P(同じ)")

    # D. 将来リターン(60 秒)とプラセボ
    b = ax[1, 0]
    q60 = F.filter(pl.col("h") == 60)
    lines(b, q60, "fwd_bp", coins, **real)
    lines(b, q60, "fwd_placebo_bp", coins, **null)
    b.axhline(0, color=INK, lw=0.8)
    b.set_title("D. 次の 60 秒の mid リターン(bp)", loc="left")
    b.set_ylabel("bp")
    b.plot([], [], color=CM, lw=1.6, ls=":", label="プラセボ(対照のレジームで切る)")
    b.legend(frameon=False, fontsize=8, loc="upper left")

    # E. 将来ボラ(60 秒)
    b = ax[1, 1]
    lines(b, q60, "fwd_rv_bp", coins, **real)
    b.set_title("E. 次の 60 秒の実現ボラ(bp)— 高ボラ側が本当に高い", loc="left")
    b.set_ylabel("bp")

    # F. 1 取引あたりの純益(最良の組み合わせ・費用控除後)
    b = ax[1, 2]
    E = pl.read_csv(D / "regime_econ.csv")
    best = (E.sort("net_bp", descending=True).group_by("coin").first()
            .sort("net_bp", descending=True))
    y = np.arange(best.height)
    b.barh(y, best["net_bp"].to_numpy(), color=C2, alpha=0.85,
           label="純益(粗利 − 往復費用)")
    b.plot(best["gross_lag1_bp"].to_numpy(), y, "o", color=INK, ms=5,
           label="粗利(判定の 1 秒後に建てる)")
    b.plot(best["rand_gross_bp"].to_numpy(), y, "x", color=C3, ms=5,
           label="無作為に同数建てた場合の粗利")
    b.axvline(0, color=INK, lw=0.8)
    b.set_yticks(y)
    b.set_yticklabels([f"xyz:{c} R{r}/{h}s" for c, r, h in
                       zip(best["coin"], best["regime"], best["h"])])
    b.invert_yaxis()
    b.set_xlabel("bp / 取引")
    b.set_title("F. 銘柄ごとの最良の組み合わせでも純益は負", loc="left")
    b.legend(frameon=False, fontsize=8, loc="upper left")
    b.grid(axis="y", visible=False)
    b.tick_params(labelsize=8)

    save(fig, "allcoins_regime.png",
         "1 秒足の 6 レジーム判定 — 状態としては実在するが、"
         "費用を引くと執行価値は残らない")


if __name__ == "__main__":
    main()
