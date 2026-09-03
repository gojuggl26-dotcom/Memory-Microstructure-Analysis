r"""README.md の LaTeX が GitHub で描画できる形かを機械的に検査する。

【なぜ要るか — 実際に起きた 3 つの事故】

  (i)  bash のヒアドキュメント経由で LaTeX を書いたとき `\\` が潰れ、
       `\f` `\r` `\n` `\a` `\b` `\t` が**制御文字**として解釈されて数式が壊れた。
       制御文字は画面に出ないので目視では気づけない。バイト列で検査する。

  (ii) **GitHub の Markdown は `\` + ASCII 記号を「バックスラッシュエスケープ」として
       先に食う。** CommonMark のエスケープ対象は  !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ 。
       したがって数式中の `\;` `\,` `\!` `\{` `\}` は KaTeX に届く前に
       `;` `,` `!` `{` `}` になる。実際に `\left\{` が `\left{` になり
       "Missing or unrecognized delimiter for \left" が出た。
       → **記号の前のバックスラッシュを数式内で使ってはいけない。**
          区切りは `\lbrace` / `\rbrace` / `\lvert` / `\rvert` を使い、
          空白調整(`\,` `\;` `\!`)は諦める。

  (iii) GitHub の KaTeX は一部のマクロを**禁止**している。
        `\operatorname` は "The following macros are not allowed: operatorname" になる。
        → `\mathrm` を使う。

【GitHub のブロック数式の要件】
  `$$` を**それ自身の行**に置き、その間に本文を書く。

【使い方】  uv run python scripts/check_readme_math.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _mdmath import count_dollars, inline_spans  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# 既定は README.md。引数でほかの .md も検査できる。
TARGET = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "README.md"

# (ii) Markdown がエスケープとして食う文字。数式内でこの前に `\` を置けない。
ESCAPABLE = set("!\"#$%&'()*+,-./:;<=>?@[]^_`{|}~\\")

# (iii) GitHub の KaTeX が拒否するマクロ
FORBIDDEN = {"operatorname", "operatorname*", "def", "gdef", "edef", "xdef",
             "let", "futurelet", "newcommand", "renewcommand", "providecommand",
             "includegraphics", "url", "href"}

# (i) ヒアドキュメント事故の残骸。
#     `\frac` が制御文字+`rac` になった痕跡を探す。ただし `rac{` は正しい `\frac{`
#     の部分文字列でもあるので、**直前が英字でもバックスラッシュでもない**ときだけ
#     残骸と判定する(否定後読み)。これが無いと全件が誤検出になる。
BROKEN = [r"(?<![A-Za-z\\])" + p for p in
          [r"rac\{", r"ight[\]\)]", r"ightbrace", r"vert\}", r"brace\}",
           r"rg\\min", r"eta\^", r"op\}"]]

# KaTeX で使う想定のコマンド。ここに無いものは目視確認を促す(NG ではない)
KNOWN = {
    "frac", "sum", "int", "in", "lvert", "rvert", "lbrace", "rbrace",
    "left", "right", "mathrm", "hat", "arg", "min", "max", "lambda",
    "beta", "top", "Omega", "tr", "qquad", "quad", "times", "approx",
    "ne", "le", "ge", "dots", "cdot", "sigma", "rho", "bar", "to", "log",
    "omega", "pi", "Omega", "alpha", "mu", "nu", "epsilon", "varepsilon",
    "gt", "lt", "amp", "sqrt", "hat", "sum", "prod",
}


def math_spans(lines: list[str]) -> list[tuple[int, str, str]]:
    """(行番号, 種別, 数式本体) を返す。インラインコード内の $ は数式にしない。"""
    out, inblock, incode = [], False, False
    for i, ln in enumerate(lines, 1):
        if ln.strip().startswith("```"):
            incode = not incode
            continue
        if incode:
            continue
        if ln.strip() == "$$":
            inblock = not inblock
            continue
        if inblock:
            out.append((i, "block", ln))
        else:
            for m in inline_spans(ln):
                body = ln[m.start() + 1:m.end() - 1]
                # ★$`...`$ はコードスパンなので、CommonMark の規則により
                #   バックスラッシュエスケープが処理されない。この形は
                #   「エスケープを食われる」検査の対象外にする(kind で分ける)。
                if len(body) >= 2 and body[0] == "`" and body[-1] == "`":
                    out.append((i, "code", body[1:-1]))
                else:
                    out.append((i, "inline", body))
    return out


def main() -> int:
    raw = TARGET.read_bytes()
    ng = 0

    # --- (i) 制御文字 ------------------------------------------------------
    bad = [(i, b) for i, b in enumerate(raw) if b < 0x20 and b not in (0x09, 0x0A, 0x0D)]
    if bad:
        ng += 1
        print(f"NG 制御文字 {len(bad)} 件: {bad[:10]}")
        for off, b in bad[:5]:
            print(f"     @{off} 0x{b:02x}  ...{raw[max(0, off - 45):off + 45]!r}")
    else:
        print("OK 制御文字なし")

    if raw.count(b"\r\n"):
        ng += 1
        print(f"NG CRLF が {raw.count(chr(13).encode() + chr(10).encode())} 行")
    else:
        print("OK 改行は LF のみ")

    lines = raw.decode("utf-8").split("\n")
    text = "\n".join(lines)

    # --- $$ が単独行か / 対になっているか ------------------------------------
    solo = [i for i, ln in enumerate(lines, 1) if ln.strip() == "$$"]
    inline_dd = [i for i, ln in enumerate(lines, 1)
                 if "$$" in ln and ln.strip() != "$$"]
    if inline_dd:
        ng += 1
        print(f"NG $$ が単独行でない: 行 {inline_dd}")
        for i in inline_dd[:5]:
            print(f"     L{i}: {lines[i - 1][:90]}")
    elif len(solo) % 2:
        ng += 1
        print(f"NG $$ の数が奇数({len(solo)})— 開閉が対でない")
    else:
        print(f"OK $$ は全て単独行・対も揃っている({len(solo) // 2} ブロック)")

    # --- 行内 $ の対応(ブロック外の各行で偶数個) ---------------------------
    inblock, incode, odd = False, False, []
    for i, ln in enumerate(lines, 1):
        if ln.strip().startswith("```"):
            incode = not incode
            continue
        if ln.strip() == "$$":
            inblock = not inblock
            continue
        if not inblock and not incode and count_dollars(ln) % 2:
            odd.append(i)
    if odd:
        ng += 1
        print(f"NG 行内 $ が奇数個の行: {odd}")
        for i in odd[:5]:
            print(f"     L{i}: {lines[i - 1][:90]}")
    else:
        print("OK 行内 $ は全行で偶数個")

    spans = math_spans(lines)

    # --- (ii) ★記号の前のバックスラッシュ ------------------------------------
    hits = []
    for ln, kind, m in spans:
        if kind == "code":          # $`...`$ は食われない(上記のとおり)
            continue
        for mo in re.finditer(r"\\(.)", m):
            if mo.group(1) in ESCAPABLE:
                hits.append((ln, kind, mo.group(0), m.strip()[:70]))
    if hits:
        ng += 1
        print(f"NG Markdown に食われるエスケープ {len(hits)} 件"
              f"(数式中の \\ + 記号は KaTeX に届かない)")
        for ln, kind, tok, ctx in hits[:8]:
            print(f"     L{ln} [{kind}] {tok!r}  :  {ctx}")
    else:
        print("OK 数式中に \\ + 記号 の並びなし")

    # --- (iv) ★数式内の裸の < > & ---------------------------------------------
    # GitHub は数式内の < > & を HTML エスケープするが、文脈によっては
    # **二重エスケープ**して `&amp;gt;` を KaTeX に渡す(実測)。そうなると
    # 不等号が「&gt;」という文字列として出る。\lt \gt \amp を使うこと。
    raw_cmp = [(ln, kind, c, m.strip()[:70])
               for ln, kind, m in spans for c in "<>&" if c in m]
    if raw_cmp:
        ng += 1
        print(f"NG 数式内に裸の < > & が {len(raw_cmp)} 件"
              f"(二重エスケープされる。\\lt \\gt \\amp を使う)")
        for ln, kind, c, ctx in raw_cmp[:8]:
            print(f"     L{ln} [{kind}] {c!r}  :  {ctx}")
    else:
        print("OK 数式内に裸の < > & なし")

    # --- (iii) 禁止マクロ ----------------------------------------------------
    used = set()
    for _, _, m in spans:
        used |= set(re.findall(r"\\([A-Za-z]+)", m))
    bad_macro = sorted(used & FORBIDDEN)
    if bad_macro:
        ng += 1
        print(f"NG GitHub が禁止しているマクロ: {bad_macro}")
    else:
        print("OK 禁止マクロなし")

    # --- (i) 壊れた残骸 ------------------------------------------------------
    残 = [m.group(0) for p in BROKEN for m in re.finditer(p, text)]
    if 残:
        ng += 1
        print(f"NG ヒアドキュメント事故の残骸: {残[:8]}")
    else:
        print("OK 既知の壊れた綴りなし")

    unknown = sorted(used - KNOWN)
    if unknown:
        print(f"?? 許可リストに無いコマンド(要目視): {unknown}")
    else:
        print(f"OK コマンド {len(used)} 種すべて既知")

    nb = len(solo) // 2
    ni = sum(1 for _, k, _ in spans if k == "inline")
    print(f"\n数式: ブロック {nb} / 行内 {ni}")
    print("=== 合格 ===" if ng == 0 else f"=== {ng} 項目 NG ===")
    return 1 if ng else 0


if __name__ == "__main__":
    sys.exit(main())
