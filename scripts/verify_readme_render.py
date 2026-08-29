r"""README.md の数式が GitHub を「通り抜ける」かを実測で検証する。

【何を確かめるか】
  check_readme_math.py は静的検査(書式)。こちらは**実際に GitHub の
  Markdown 変換器に投げて、数式の文字列が変化しないこと**を確かめる。

  GitHub の Markdown は `\` + ASCII 記号をバックスラッシュエスケープとして
  消費する。数式がその影響を受けると KaTeX に届く前に壊れる
  (実測: `\left\{` -> `\left{` で "Missing or unrecognized delimiter for \left")。
  変換後の HTML に元の LaTeX がそのまま残っていれば、この経路では壊れない。

  注意: /markdown API は KaTeX を走らせない(数式として組版はしない)。
  ここで検証しているのは**エスケープ処理を通り抜けるか**だけで、
  組版できるかは §2 のコマンド照合で見る。

【使い方】 PYTHONIOENCODING=utf-8 python scripts/verify_readme_render.py
"""
from __future__ import annotations

import html
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _mdmath import inline_spans  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TARGET = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "README.md"


def spans(lines: list[str]) -> list[tuple[str, str]]:
    """README の数式を出現順に返す。インラインコード内の $ は数式にしない。"""
    out, inblock, incode, buf = [], False, False, []
    for ln in lines:
        if ln.strip().startswith("```"):
            incode = not incode
            continue
        if incode:
            continue
        if ln.strip() == "$$":
            if inblock:
                out.append(("block", "\n".join(buf)))
                buf = []
            inblock = not inblock
            continue
        if inblock:
            buf.append(ln)
        else:
            out += [("inline", ln[m.start() + 1:m.end() - 1])
                    for m in inline_spans(ln)]
    return out


def gh_markdown(text: str) -> str:
    payload = json.dumps({"text": text, "mode": "gfm"}, ensure_ascii=False)
    p = subprocess.run(
        ["gh", "api", "-X", "POST", "markdown", "--input", "-"],
        input=payload.encode("utf-8"), capture_output=True,
        env={**__import__("os").environ, "MSYS_NO_PATHCONV": "1"})
    if p.returncode != 0:
        print("gh api 失敗:", p.stderr.decode("utf-8", "replace")[:300])
        sys.exit(2)
    return p.stdout.decode("utf-8")


def main() -> int:
    src = TARGET.read_text(encoding="utf-8")
    ms = spans(src.split("\n"))
    print(f"数式 {len(ms)} 件(ブロック {sum(1 for k,_ in ms if k=='block')} / "
          f"行内 {sum(1 for k,_ in ms if k=='inline')})\n")

    out = html.unescape(gh_markdown(src))

    ng = []
    for kind, m in ms:
        # ブロックは複数行。行ごとに、意味のある断片が残っているかを見る
        for frag in [x.strip() for x in m.split("\n") if x.strip()]:
            if frag not in out:
                ng.append((kind, frag))

    if ng:
        print(f"NG GitHub の変換で消えた/変化した断片 {len(ng)} 件:")
        for kind, f in ng[:12]:
            print(f"   [{kind}] {f[:100]}")
    else:
        print("OK 全数式が GitHub の Markdown 変換をそのまま通り抜けた")
        print("   (= \\ + 記号 のエスケープ消費は起きていない)")

    # --- ★より強い検査: GitHub が「数式として認識」したか ---------------------
    # 素通りしただけでは足りない。地の文として通り抜けた可能性があるので、
    # <math-renderer> に入ったかを数える。入っていなければ生テキストで表示される。
    tags = [html.unescape(t) for t in
            re.findall(r"<math-renderer[^>]*>(.*?)</math-renderer>", out, re.S)]
    disp = len(re.findall(r"js-display-math", out))
    inl = len(re.findall(r"js-inline-math", out))
    print(f"\n認識: math-renderer {len(tags)} 個(ブロック {disp} / 行内 {inl})"
          f" 対 README の数式 {len(ms)} 件")

    # ★順序どおりに突き合わせる。部分文字列検索だと、同じ式が別の行で成功して
    #   いるときに失敗を見落とす。認識された列と期待する列を先頭から比較する。
    got = [re.sub(r"^\$\$?|\$\$?$", "", t.strip()).strip() for t in tags]
    want = [" ".join(m.split()) for _, m in ms]
    got = [" ".join(g.split()) for g in got]
    miss, gi = [], 0
    for (kind, _), w in zip(ms, want):
        if gi < len(got) and got[gi] == w:
            gi += 1
        else:
            miss.append((kind, w))
    if miss:
        ng += miss
        print(f"NG 数式として認識されなかった {len(miss)} 件"
              f"(= ページ上に $...$ の生テキストで出る):")
        for kind, f in miss:
            # 直前の文字を突き止める(これが原因になる)
            i = src.find("$" + f.split()[0][:12]) if f else -1
            prev = src[i - 1] if i > 0 else "?"
            print(f"   [{kind}] 直前={prev!r}  {f[:80]}")
    else:
        print("OK 全数式が math-renderer に入った(= 生テキスト表示にはならない)")

    # 対照: わざと壊れる書き方が本当に壊れることを確認する(検査の妥当性確認)
    probe = r"$$\left\{ a \;+\; b \right\}$$"
    po = html.unescape(gh_markdown(probe))
    broke = r"\left\{" not in po
    print(f"\n帰無対照 — わざと `\\left\\{{` `\\;` を使った式: "
          f"{'期待どおり壊れた' if broke else '★壊れなかった(仮説の見直しが必要)'}")
    if not broke:
        print("   → エスケープ消費が原因という診断が誤っている可能性がある")
    print(f"   変換後: {po.strip()[:160]}")

    return 1 if ng else 0


if __name__ == "__main__":
    sys.exit(main())
