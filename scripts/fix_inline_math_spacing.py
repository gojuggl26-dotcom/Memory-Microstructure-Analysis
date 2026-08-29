r"""行内数式の開始 `$` の直前に半角空白を入れて、GitHub が数式と認識できる形にする。

【なぜ必要か — 実測で確定した規則】
  probe_inline_math.py の対照実験より、GitHub が `$...$` を数式と認識するかは
  **開始 `$` の直前の文字だけ**で決まる(閉じ側の直後は何が来ても影響しない)。

      通る : 行頭 / 半角空白 / 全角開き括弧 ( / アスタリスク *
      通らない: 漢字・かな / 読点 、 / ASCII カンマ , / 中黒 ・

  日本語の文中では読点や漢字のすぐ後ろに数式を置きがちなので、
  そのままだと `$z = 9.2$` がページ上に**生テキストで表示される**。

【この修正がすること】
  開始 `$` の直前が許可された文字でなければ、半角空白を 1 つ挿入する。
  数式の中身には一切触れない(バックスラッシュ事故を繰り返さないため)。

【使い方】 PYTHONIOENCODING=utf-8 python scripts/fix_inline_math_spacing.py [--apply]
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _mdmath import inline_spans  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
_ARGS = [x for x in sys.argv[1:] if not x.startswith("--")]
TARGET = Path(_ARGS[0]) if _ARGS else ROOT / "README.md"

OK_BEFORE = set(" \t(*")   # 半角空白 / タブ / 全角開き括弧 / 強調のアスタリスク


def main() -> int:
    apply = "--apply" in sys.argv
    lines = TARGET.read_text(encoding="utf-8").split("\n")

    out, inblock, incode = [], False, False
    n_fix = 0
    before = Counter()
    for ln in lines:
        s = ln.strip()
        if s.startswith("```"):
            incode = not incode
            out.append(ln)
            continue
        if s == "$$":
            inblock = not inblock
            out.append(ln)
            continue
        if incode or inblock or "$" not in ln:
            out.append(ln)
            continue

        # 行内数式を左から順に対にして、開始 $ の位置だけ集める
        # (インラインコード内の $ は除外。位置はマスクしても保存される)
        starts = [m.start() for m in inline_spans(ln)]
        new, prev = [], 0
        for i in starts:
            c = ln[i - 1] if i > 0 else ""
            before[c or "<行頭>"] += 1
            new.append(ln[prev:i])
            if i > 0 and c not in OK_BEFORE:
                new.append(" ")
                n_fix += 1
            prev = i
        new.append(ln[prev:])
        out.append("".join(new))

    print("開始 $ の直前の文字(出現数):")
    for c, n in before.most_common():
        mark = "OK" if (c == "<行頭>" or c in OK_BEFORE) else "NG"
        print(f"  {mark}  {c!r:>10}  {n:>3}")
    print(f"\n空白を挿入する箇所: {n_fix} 件")

    if apply and n_fix:
        TARGET.write_text("\n".join(out), encoding="utf-8", newline="\n")
        print(f"-> {TARGET} に書き込んだ")
    elif not apply:
        print("(--apply を付けると書き込む)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
