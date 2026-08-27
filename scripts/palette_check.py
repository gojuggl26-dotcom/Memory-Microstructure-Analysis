"""配色の計算可能な検査(目視しない).

dataviz スキルの `scripts/validate_palette.js` の Python 移植。
node がこの環境に無いため、同じ定数・同じ変換で計算する。

検査するのは色だけから測れる 5 項目:

    2  明度帯     OKLCH の L がモードの帯に入っているか
    3  彩度の下限 OKLCH の C が下限以上か(下回ると灰色に見える)
    4  CVD 分離   protan / deutan をかけた OKLab ΔE(×100)。隣接ペア
    4b 通常視の下限 未シミュレートの ΔE。ここが低いと色覚が正常でも見分けられない
    5  背景との対比 WCAG コントラスト比

CVD 変換は Machado, Oliveira & Fernandes (2009) の severity 1.0。
閾値はこのシミュレーションに合わせて較正されているので、
別のモデル(Viénot 1999 等)に差し替えるなら閾値も較正し直すこと。
"""

from __future__ import annotations

import math

BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}   # OKLCH L
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0     # OKLab ΔE×100、min(protan, deutan)、隣接ペア
NORMAL_FLOOR = 15.0                  # 未シミュレートの最悪ペア(ここは hard fail)
CONTRAST_MIN = 3.0
DEFAULT_SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}

MACHADO = {
    "protan": ((0.152286, 1.052583, -0.204868),
               (0.114503, 0.786281, 0.099216),
               (-0.003882, -0.048116, 1.051998)),
    "deutan": ((0.367322, 0.860646, -0.227968),
               (0.280085, 0.672501, 0.047413),
               (-0.011820, 0.042940, 0.968881)),
    "tritan": ((1.255528, -0.076749, -0.178779),
               (-0.078411, 0.930809, 0.147602),
               (0.004733, 0.691367, 0.303900)),
}


def _srgb(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.strip().lstrip("#")
    if len(h) != 6 or any(c not in "0123456789abcdefABCDEF" for c in h):
        raise ValueError(f"16 進 6 桁の色ではない: {hex_color!r}")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _lin(hex_color: str) -> tuple[float, float, float]:
    return tuple(_to_linear(c) for c in _srgb(hex_color))


def _oklab_from_lin(rgb) -> tuple[float, float, float]:
    r, g, b = rgb
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    )


def oklch(hex_color: str) -> tuple[float, float]:
    """OKLCH の (L, C) を返す."""
    L, a, b = _oklab_from_lin(_lin(hex_color))
    return L, math.hypot(a, b)


def relative_luminance(hex_color: str) -> float:
    r, g, b = _lin(hex_color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((relative_luminance(a), relative_luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _simulate(hex_color: str, kind: str):
    r, g, b = _lin(hex_color)
    M = MACHADO[kind]
    return tuple(
        min(1.0, max(0.0, M[i][0] * r + M[i][1] * g + M[i][2] * b)) for i in range(3)
    )


def delta_e(h1: str, h2: str, kind: str | None = None) -> float:
    """OKLab のユークリッド距離 ×100。kind=None で未シミュレート(通常視)."""
    a = _oklab_from_lin(_simulate(h1, kind) if kind else _lin(h1))
    b = _oklab_from_lin(_simulate(h2, kind) if kind else _lin(h2))
    return 100 * math.dist(a, b)


def validate(palette, *, mode: str = "light", surface: str | None = None,
             pairs: str = "adjacent") -> dict:
    """配色を検査して結果を返す.

    pairs="adjacent" は折れ線・棒など系列が順に並ぶ形式、
    "all" は散布図・地図など全ペアが同時に見える形式で使う。
    """
    surface = surface or DEFAULT_SURFACE[mode]
    lo, hi = BAND[mode]
    rows, failures, warnings = [], [], []

    for i, c in enumerate(palette):
        L, C = oklch(c)
        cr = contrast(c, surface)
        row = {"slot": i, "hex": c, "L": round(L, 4), "C": round(C, 4),
               "contrast": round(cr, 2), "status": "PASS"}
        if not (lo <= L <= hi):
            row["status"] = "FAIL"
            failures.append(f"slot {i} {c}: L={L:.3f} が明度帯 [{lo}, {hi}] の外")
        if C < CHROMA_FLOOR:
            row["status"] = "FAIL"
            failures.append(f"slot {i} {c}: C={C:.3f} が彩度下限 {CHROMA_FLOOR} 未満")
        if cr < CONTRAST_MIN:
            if row["status"] == "PASS":
                row["status"] = "WARN"
            warnings.append(f"slot {i} {c}: 対比 {cr:.2f} < {CONTRAST_MIN}(要ラベル or 表)")
        rows.append(row)

    idx = (
        [(i, i + 1) for i in range(len(palette) - 1)]
        if pairs == "adjacent"
        else [(i, j) for i in range(len(palette)) for j in range(i + 1, len(palette))]
    )
    pair_rows = []
    for i, j in idx:
        a, b = palette[i], palette[j]
        p, d = delta_e(a, b, "protan"), delta_e(a, b, "deutan")
        cvd, normal = min(p, d), delta_e(a, b)
        status = "PASS"
        if normal < NORMAL_FLOOR:
            status = "FAIL"
            failures.append(f"{a}/{b}: 通常視 ΔE={normal:.1f} < {NORMAL_FLOOR}")
        if cvd < CVD_FLOOR:
            status = "FAIL"
            failures.append(f"{a}/{b}: CVD ΔE={cvd:.1f} < 下限 {CVD_FLOOR}")
        elif cvd < CVD_TARGET:
            status = "WARN" if status == "PASS" else status
            warnings.append(f"{a}/{b}: CVD ΔE={cvd:.1f} < 目標 {CVD_TARGET}(二重符号化が必須)")
        pair_rows.append({"pair": f"{i}-{j}", "a": a, "b": b,
                          "protan": round(p, 1), "deutan": round(d, 1),
                          "tritan": round(delta_e(a, b, "tritan"), 1),
                          "normal": round(normal, 1), "status": status})

    return {"mode": mode, "surface": surface, "pairs": pairs,
            "slots": rows, "pairs_checked": pair_rows,
            "failures": failures, "warnings": warnings,
            "ok": not failures}
