r"""README の数式を扱う 3 つのスクリプトで共有する抽出処理。

【なぜ切り出したか】
  抽出規則を 3 か所に書いたところ、`$` を含む**インラインコード**
  (例: バッククォートで囲った `$`)を数式と誤認して偽陽性を出した。
  規則は 1 か所に置き、3 スクリプトが同じ判定を使う。

【マスクするもの】
  - インラインコード `...`  … 中の `$` は数式にならない
  - フェンスコードブロック  … 呼び出し側で行単位に扱う
  - ブロック数式 $$ ... $$  … 呼び出し側で行単位に扱う

  マスクは**同じ長さの別文字に置換**する。位置がずれると
  「開始 $ の直前の文字」を見る処理が壊れるため。
"""
from __future__ import annotations

import re

# インラインコード。バッククォートの数が一致する対を優先して長い方から食う。
_CODE = re.compile(r"(`+)(?:(?!\1).)*?\1")


def mask_code(line: str) -> str:
    """インラインコード span を同じ長さの 'x' に置換した行を返す。"""
    return _CODE.sub(lambda m: "x" * len(m.group(0)), line)


def inline_spans(line: str) -> list[re.Match]:
    """行内数式 $...$ の Match を左から順に返す(コード内は除外)。"""
    return list(re.finditer(r"\$[^$\n]+\$", mask_code(line)))


def count_dollars(line: str) -> int:
    """コード内を除いた `$` の数。奇偶の検査用。"""
    return mask_code(line).count("$")
