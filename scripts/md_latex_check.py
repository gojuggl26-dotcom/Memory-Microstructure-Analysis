"""変換した LaTeX の構文検算。GitHub 上での描画は確認できないので、機械的に見る。

見るもの:
  1. $ の対応が取れているか(フェンス外で偶数個か)
  2. 各数式の波括弧が閉じているか
  3. 使っているコマンドが、GitHub の数式描画(KaTeX)で通る範囲か
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

ROOT = Path("C:/Users/ii562/Downloads/Memory")

# KaTeX で確実に通るもの。ここに無いコマンドが出たら目視する
ALLOWED = {
    "log", "sum", "lceil", "rceil", "mathrm", "frac", "tfrac", "beta", "rho",
    "sigma", "lambda", "alpha", "varepsilon", "Delta", "delta", "infty",
    "times", "cdot", "approx", "pm", "ge", "le", "ne", "hat", "bar",
    "textstyle", "left", "right", "corr", "to",
    "lvert", "rvert", "tau",       # 既存の数式で使われている(KaTeX で通る)
}

FENCE = re.compile(r"```.*?```", re.S)
# 前後が \ のものは通貨のエスケープ(\$60)なので数式の区切りにしない
INLINE = re.compile(r"(?<![$\\])\$([^$\n]+?)(?<!\\)\$(?!\$)")

n_eq = 0
problems: list[str] = []
unknown: dict[str, int] = {}

for f in sorted(ROOT.glob("*.md")):
    s = f.read_text(encoding="utf-8")
    body = FENCE.sub("", s)

    dollars = len(re.findall(r"(?<!\\)\$", body))
    if dollars % 2:
        problems.append(f"{f.name}: $ が奇数 ({dollars}) — 対応が崩れている")

    for m in INLINE.finditer(body):
        n_eq += 1
        e = m.group(1)
        if e.count("{") != e.count("}"):
            problems.append(f"{f.name}: 波括弧が不一致  {e[:60]}")
        # 半開区間 [a, b) や (a, b] は括弧が非対称でも正しい記法なので除く
        if e.count("(") != e.count(")") and not re.match(r"^[\[(].*[\])]$", e.strip()):
            problems.append(f"{f.name}: 丸括弧が不一致  {e[:60]}")
        for c in re.findall(r"\\([a-zA-Z]+)", e):
            if c not in ALLOWED:
                unknown[c] = unknown.get(c, 0) + 1

print(f"インライン数式 {n_eq} 箇所")
if unknown:
    print("要目視のコマンド:", ", ".join(f"\\{k}({v})" for k, v in sorted(unknown.items())))
else:
    print("使用コマンドはすべて KaTeX で通る範囲")
if problems:
    print(f"\n問題 {len(problems)} 件:")
    for p in problems[:20]:
        print("  ", p)
else:
    print("括弧・$ の対応に問題なし")
