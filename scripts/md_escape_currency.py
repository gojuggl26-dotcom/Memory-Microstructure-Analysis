"""通貨の $ をエスケープして、数式として描画されるのを防ぐ。

このリポジトリは金額を $60 や ≈$1.50 のように書く。GitHub は $...$ を
数式として拾うため、1 行に通貨の $ が複数あると

    EC2(stopped)$25.12 | S3 ≈$1.50 | 累計 ≈$26.62 / 予算 $60(停止閾値 $48)

の "25.12 | S3 ≈" などが数式として描画されてしまう。

規則: フェンスとインラインコードの外で、直後が数字の $ は通貨とみなして \\$ にする。
      このリポジトリの数式は $ の直後が英字かバックスラッシュで始まるので、
      数字始まりと衝突しない(全 58 箇所で確認済み)。

使い方:
    uv run python scripts/md_escape_currency.py --check
    uv run python scripts/md_escape_currency.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path("C:/Users/ii562/Downloads/Memory")
FENCE = re.compile(r"```.*?```", re.S)
INLINE = re.compile(r"`[^`\n]*`")
CURRENCY = re.compile(r"(?<!\\)\$(?=\d)")


def protected_spans(text: str):
    spans = [m.span() for m in FENCE.finditer(text)]

    def in_fence(i: int) -> bool:
        return any(a <= i < b for a, b in spans)

    for m in INLINE.finditer(text):
        if not in_fence(m.start()):
            spans.append(m.span())
    return sorted(spans)


def convert(text: str) -> tuple[str, int]:
    spans = protected_spans(text)
    out, last, n = [], 0, 0
    for a, b in spans:
        seg = text[last:a]
        new, k = CURRENCY.subn(r"\\$", seg)
        out.append(new); n += k
        out.append(text[a:b])
        last = b
    new, k = CURRENCY.subn(r"\\$", text[last:])
    out.append(new); n += k
    return "".join(out), n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    total = 0
    files = 0
    for f in sorted(ROOT.glob("*.md")):
        src = f.read_text(encoding="utf-8")
        dst, n = convert(src)
        if n:
            total += n; files += 1
            print(f"  {f.name:<36} {n:>3} 個")
            if not a.check:
                f.write_text(dst, encoding="utf-8")
    print(f"\n{'検査' if a.check else 'エスケープ'}: {files} 本 / 合計 {total} 個")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
