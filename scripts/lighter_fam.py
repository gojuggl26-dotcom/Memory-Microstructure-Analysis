r"""特徴量名 → 系統(接頭辞で機械的に決まる。結果を見る前に確定)。"""
from __future__ import annotations

FAMS = [
    (1, "① 価格・リターン・スプレッド", ("px_", "ret_", "rv_", "spr_", "path_", "range_")),
    (2, "② OBI(板の不均衡)", ("obi_",)),
    (3, "③ 深さ", ("dep_",)),
    (4, "④ microprice", ("mic_",)),
    (5, "⑤ 板の形状", ("shp_",)),
    (6, "⑥ 価格の隙間", ("gap_",)),
    (7, "⑦ 指値フロー", ("flw_",)),
    (8, "⑧ 取消", ("cxl_",)),
    (9, "⑨ OFI", ("ofi_",)),
    (10, "⑩ 約定フロー", ("trd_",)),
    (11, "⑪ 活動・品質", ("act_",)),
    (12, "⑫ 掛け合わせ", ("x_",)),
]
NAME = {k: n for k, n, _ in FAMS}
COLORS = {1: "#6b7280", 2: "#dc2626", 3: "#94a3b8", 4: "#7c3aed",
          5: "#2563eb", 6: "#0891b2", 7: "#65a30d", 8: "#16a34a",
          9: "#ea580c", 10: "#be123c", 11: "#a16207", 12: "#0d9488"}


def family(name: str) -> int:
    for k, _, prefs in FAMS:
        if name.startswith(prefs):
            return k
    return 0
