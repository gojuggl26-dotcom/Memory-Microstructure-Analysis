"""Markdown の体裁を正規化する。

やること:
  1. 散文中のアスタリスクを全部消す(太字 `**...**`、斜体 `*...*`、単独の `*`)
  2. ただし コードブロック(```)とインラインコード(`...`)の中は触らない
     — そこの `*` は `data/*.json` のような**リテラルのパス表記**で、
       消すと文書が壊れるため

`--check` で消し残しと保護対象の件数だけ出す(書き換えない)。

使い方:
    uv run python scripts/md_normalize.py --check
    uv run python scripts/md_normalize.py
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

ROOT = Path("C:/Users/ii562/Downloads/Memory")

FENCE = re.compile(r"```.*?```", re.S)
INLINE = re.compile(r"`[^`\n]*`")


def _split_protected(text: str):
    """(素の断片, 保護断片) の交互列へ分解する。保護断片はそのまま戻す。"""
    parts, last = [], 0
    spans = [m.span() for m in FENCE.finditer(text)]
    # フェンスの外だけでインラインコードを探す
    def in_fence(i):
        return any(a <= i < b for a, b in spans)
    for m in INLINE.finditer(text):
        if not in_fence(m.start()):
            spans.append(m.span())
    spans.sort()
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    for a, b in merged:
        parts.append(("text", text[last:a]))
        parts.append(("code", text[a:b]))
        last = b
    parts.append(("text", text[last:]))
    return parts


def strip_stars(text: str) -> str:
    """散文からアスタリスクを除く。太字・斜体の記号だけを落とし中身は残す。"""
    t = re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", text, flags=re.S)
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t, flags=re.S)
    t = re.sub(r"\*(.+?)\*", r"\1", t, flags=re.S)
    t = t.replace("*", "")            # 対で閉じていない残り
    return t


def process(path: Path, write: bool):
    src = path.read_text(encoding="utf-8")
    out, kept = [], 0
    for kind, chunk in _split_protected(src):
        if kind == "code":
            kept += chunk.count("*")
            out.append(chunk)
        else:
            out.append(strip_stars(chunk))
    dst = "".join(out)
    changed = dst != src
    if write and changed:
        path.write_text(dst, encoding="utf-8")
    return src.count("*"), dst.count("*"), kept, changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    files = sorted(ROOT.glob("*.md"))
    b_tot = a_tot = k_tot = 0
    n_changed = 0
    for f in files:
        before, after, kept, changed = process(f, write=not a.check)
        b_tot += before; a_tot += after; k_tot += kept
        n_changed += changed
        if before and (a.check or changed):
            print(f"  {f.name:<38} {before:>6} → {after:>4}  (コード内で保護 {kept})")
    print(f"\n{'検査' if a.check else '書換'}: {len(files)} 本 / 変更 {n_changed} 本")
    print(f"  アスタリスク {b_tot:,} → {a_tot:,}(保護 {k_tot} を含む)")
    if a_tot != k_tot:
        print(f"  ★保護対象以外に {a_tot - k_tot} 個残っている — 要確認")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
