r"""GitHub で描画されない数式記法を、MAINTENANCE.txt の規約形へ直す。

【なぜ要るか】
GitHub の Markdown は **KaTeX に渡す前にバックスラッシュのエスケープを食う**。
`$...$` / `$$...$$` で書くと `\,` `\;` `\!` `\{` `\#` が
`,` `;` `!` `{` `#` に化け、数式が壊れて生の LaTeX が表示される。

規約形はこれを構造的に避ける。

    行内      $`...`$        … 中身が**コードスパン**なのでエスケープを食われない
    別行立て  ```math フェンス … 中身が**コードブロック**なので同じく食われない

さらに、数式中の裸の `<` `>` は二重エスケープされるので `\lt` `\gt` にする。
金額の `$` は数式の開始と誤認されるので `\$` にする(表の行が丸ごと
数式に飲まれて崩れる事故が実際にあった)。

【この script がすること】
    1. 金額の `$`(数字が続くもの)を `\$` にする
    2. `$$ ... $$` を ```math フェンスにする
    3. 素の行内 `$...$` を $`...`$ にする(LaTeX らしい記号を含むものだけ)
    4. 数式の中の裸の `<` `>` を `\lt` `\gt` にする
    5. CRLF を LF にする

**コードフェンス(```)の中身には触らない。** bash の `$VAR` を壊さないため。

【使い方】
    uv run python scripts/fix_md_math.py reports/hyperliquid/MU/*.md
    uv run python scripts/fix_md_math.py --check reports/**/*.md   # 書き換えずに件数だけ

直したあとは必ず `scripts/check_readme_math.py` を通すこと。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# 数式とみなす条件。バックスラッシュ等を含むか、短い変数名であること。
# 数字で始まるもの($460 など)は金額なので数式にしない。
LATEX_HINT = re.compile(r"[\\_^{}]")
VARNAME = re.compile(r"^[A-Za-zα-ωΑ-Ω][A-Za-z0-9'’\s]{0,3}$")
DD_BLOCK = re.compile(r"\$\$(.+?)\$\$", re.S)
INLINE = re.compile(r"(?<![\\$`])\$(?!`)([^$\n]{1,200}?)(?<![\\$])\$(?!\$)")
CURRENCY = re.compile(r"(?<!\\)\$(?=[0-9])")


def split_fences(text: str):
    """``` で囲まれた部分と、それ以外に分ける。順序を保った (is_fence, 文字列)。"""
    out, buf, in_f = [], [], False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            out.append((in_f, "\n".join(buf)))
            buf = [line]
            if in_f:                       # 閉じ
                out.append((True, "\n".join(buf)))
                buf, in_f = [], False
            else:
                in_f = True
            continue
        buf.append(line)
    out.append((in_f, "\n".join(buf)))
    return [(f, s) for f, s in out if s != ""]


def fix_math_body(s: str) -> str:
    """数式の中身だけの補正。裸の < > を KaTeX が読める形にする。

    ★空白の詰め直しはしない。描画に影響しないうえ、既に規約どおりの
    ファイルまで差分だらけになって、本当の修正が埋もれるため。
    """
    s = re.sub(r"(?<![\\<>])<(?!/)", r"\\lt ", s)
    return re.sub(r"(?<![\\<>-])>", r"\\gt ", s)


def convert(text: str) -> tuple[str, dict]:
    n = dict(currency=0, block=0, inline=0, lt=0, skipped_table=0)
    parts = []
    for is_fence, seg in split_fences(text):
        if is_fence:
            parts.append(seg)
            continue

        # (1) 金額の $
        seg, k = CURRENCY.subn(r"\\$", seg)
        n["currency"] += k

        # (2) $$ ... $$ -> ```math
        def blk(m):
            body = m.group(1).strip()
            # 表の中の $$ はフェンスにできない。触らずに残して警告する
            line_start = seg.rfind("\n", 0, m.start()) + 1
            if seg[line_start:m.start()].lstrip().startswith("|"):
                n["skipped_table"] += 1
                return m.group(0)
            n["block"] += 1
            before = "" if m.start() == 0 or seg[m.start() - 1] == "\n" else "\n"
            after = "" if m.end() >= len(seg) or seg[m.end()] == "\n" else "\n"
            body2 = re.sub(r"[ \t]+", " ", fix_math_body(body)).strip()
            n["lt"] += body2.count("\\lt") + body2.count("\\gt") \
                - body.count("\\lt") - body.count("\\gt")
            return f"{before}```math\n{body2}\n```{after}"

        seg = DD_BLOCK.sub(blk, seg)

        # (3) 素の行内 $...$ -> $`...`$
        def inl(m):
            body = m.group(1)
            if not (LATEX_HINT.search(body) or VARNAME.match(body)):
                return m.group(0)                # 金額や単なる文字列は触らない
            n["inline"] += 1
            body2 = fix_math_body(body)
            n["lt"] += body2.count("\\lt") + body2.count("\\gt") \
                - body.count("\\lt") - body.count("\\gt")
            return f"$`{body2}`$"

        seg = INLINE.sub(inl, seg)

        # (4) もともと規約形だった行内数式の中の裸の < >
        def okinl(m):
            body = m.group(1)
            body2 = fix_math_body(body)
            n["lt"] += body2.count("\\lt") + body2.count("\\gt") \
                - body.count("\\lt") - body.count("\\gt")
            return f"$`{body2}`$"

        seg = re.sub(r"\$`([^`]*)`\$", okinl, seg)
        parts.append(seg)

    out = "\n".join(parts)

    # (5) 既存の ```math フェンスの中身も < > を直す
    def fence_fix(m):
        body = m.group(1)
        body2 = fix_math_body(body)
        n["lt"] += body2.count("\\lt") + body2.count("\\gt") \
            - body.count("\\lt") - body.count("\\gt")
        return f"```math\n{body2}\n```"

    out = re.sub(r"```math\n(.*?)\n```", fence_fix, out, flags=re.S)
    return out.replace("\r\n", "\n").replace("\r", "\n"), n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--check", action="store_true", help="書き換えずに件数だけ出す")
    a = ap.parse_args()
    total = dict(currency=0, block=0, inline=0, lt=0, skipped_table=0)
    changed = 0
    for f in a.files:
        p = Path(f)
        if not p.is_file():
            continue
        src = p.read_text(encoding="utf-8")
        out, n = convert(src)
        if out != src:
            changed += 1
            if not a.check:
                p.write_text(out, encoding="utf-8", newline="\n")
            print(f"  {p.name}: 金額 {n['currency']} / ブロック {n['block']} / "
                  f"行内 {n['inline']} / <> {n['lt']}"
                  + (f" / ★表内で飛ばした $$ {n['skipped_table']}"
                     if n["skipped_table"] else ""))
        for k in total:
            total[k] += n[k]
    verb = "検出" if a.check else "修正"
    print(f"\n{changed} ファイルを{verb}: 金額 {total['currency']} / "
          f"ブロック {total['block']} / 行内 {total['inline']} / <> {total['lt']}")
    if total["skipped_table"]:
        print(f"★表の中の $$ が {total['skipped_table']} 件ある。"
              f"フェンスにできないので手で直すこと", file=sys.stderr)


if __name__ == "__main__":
    main()
