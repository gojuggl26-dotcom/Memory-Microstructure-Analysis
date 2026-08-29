"""図の共通体裁。既存スクリプト(図 1〜49)が各自で持っていた設定を 1 箇所にまとめたもの。

色と rcParams の値は既存スクリプトからそのまま移しており、見た目は変わらない。
新しい図を足すたびに同じ 8 行を複製しないためだけの存在で、
既存スクリプトは触っていない(動いているものを壊さないため)。

使い方:
    from _chartstyle import C1, C2, C3, C4, CM, INK, GRID, D, CH, plt, np
"""
from __future__ import annotations
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

ROOT = Path("C:/Users/ii562/Downloads/Memory")
D, CH = ROOT / "data", ROOT / "charts"

C1, C2, C3, C4 = "#3b6fd4", "#c2410c", "#0f766e", "#7c3aed"
CM, INK, GRID = "#9aa3b2", "#1f2733", "#e3e7ee"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9, "font.family": "sans-serif",
    "font.sans-serif": ["Yu Gothic", "Meiryo", "MS Gothic", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name: str, suptitle: str | None = None) -> Path:
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=0.985)
        fig.tight_layout(rect=[0, 0, 1, 0.965])
    else:
        fig.tight_layout()
    out = CH / name
    fig.savefig(out, facecolor="white")
    print("保存:", out)
    return out
