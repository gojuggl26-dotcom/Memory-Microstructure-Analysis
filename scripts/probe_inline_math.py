r"""GitHub が行内数式をどう認識するかを、行単位の実験で切り分ける。

verify_readme_render.py で「6 件の行内数式が math-renderer に入らない」ことが
判った。原因を推測で決めず、**該当行を単独で投げて再現するか**、
さらに条件を 1 つずつ変えた対照を並べて切り分ける。
"""
from __future__ import annotations

import html
import json
import os
import re
import subprocess
import sys


def gh(text: str) -> str:
    payload = json.dumps({"text": text, "mode": "gfm"}, ensure_ascii=False)
    p = subprocess.run(["gh", "api", "-X", "POST", "markdown", "--input", "-"],
                       input=payload.encode("utf-8"), capture_output=True,
                       env={**os.environ, "MSYS_NO_PATHCONV": "1"})
    if p.returncode != 0:
        print("gh api 失敗:", p.stderr.decode("utf-8", "replace")[:200])
        sys.exit(2)
    return html.unescape(p.stdout.decode("utf-8"))


def n_math(out: str) -> int:
    return len(re.findall(r"<math-renderer", out))


CASES = [
    # --- 隣接文字を 1 つずつ変えて、どの位置の何が効くかを切り分ける -----------
    ("基準: 前後とも半角空白",        "あ $x$ い",      1),
    ("前が漢字(空白なし)",          "あ$x$ い",       1),
    ("後が漢字(空白なし)",          "あ $x$い",       1),
    ("前後とも漢字(空白なし)",      "あ$x$い",        1),
    ("★前が読点 、",                 "あ、$x$ い",     1),
    ("★後が読点 、",                 "あ $x$、い",     1),
    ("前が読点+空白",                "あ、 $x$ い",    1),
    ("後が空白+読点",                "あ $x$ 、い",    1),
    ("★後が句点 。",                 "あ $x$。い",     1),
    ("前が全角開き括弧 (",           "あ($x$ い",     1),
    ("★後が全角閉じ括弧 )",          "あ $x$)い",     1),
    ("前が ASCII カンマ",             "a,$x$ b",        1),
    ("後が ASCII カンマ",             "a $x$,b",        1),
    ("前が全角中黒 ・",               "あ・$x$ い",     1),
    # --- 実際に失敗した行と、その修正案 --------------------------------------
    ("実際に失敗した行",              "(中央 **4.33 倍**、$z = 9.2$、$p = 1$)。", 2),
    ("修正案: 読点の後に空白",        "(中央 **4.33 倍**、 $z = 9.2$、 $p = 1$)。", 2),
]


def main() -> int:
    print(f"{'期待':>4} {'実測':>4}  説明")
    print("-" * 72)
    for desc, txt, exp in CASES:
        got = n_math(gh(txt))
        mark = "OK " if got == exp else "NG "
        print(f"{mark}{exp:>3} {got:>4}  {desc}")
        if got != exp:
            print(f"          入力: {txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
