r"""イベント時間の Pressure と将来リターンの関係を OLS / GLS で測る(全 188 市場)。

=============================================================================
定義
=============================================================================
  各市場のイベント列(板が両側非空かつ非クロスの時点のみ)を t = 0,1,2,… とする。

    注文流入量_t(s)  = Q_s(t)
    OFM_t(s)         = Q_s(t) − Q_s(t−1)
    Pressure_t(s)    = OFM_t(s) / bq_{o(s)}(t)      o(s) = 反対側

  ★「注文流入量」の読みを 2 通り併走させる。差分を取っている以上これは**残高量**
    であって流量ではないが、どの残高かで 2 通りある:

      (A) best  Q_s = bq_s    その側の**最良気配 1 レベル**の数量。
          分母(反対側の最良数量)と対称になる自然な読み。**主**とする。
          ただし実測では **8 割超のイベントで OFM が厳密に 0** になる
          (大半のイベントは最良レベルに触れない)。
      (B) depth Q_s = depth_s その側の**全レベルの合計**数量。
          ほぼ毎イベント動くのでゼロ膨張が無い。**副**として併記する。

    どちらが依頼の意図かは一意に決まらないので、片方を黙って選ばず両方出す。

  Boros では rate 空間で LONG 側が bid、SHORT 側が ask に相当する
  (spread = best_short − best_long > 0)。したがって

    pressure_diff_t = Pressure_t(long) − Pressure_t(short)

  は「買い圧 − 売り圧」に相当し、**将来の implied APR を押し上げる向き(正)**が
  事前予測である。片側だけを見る Pressure(long) も正、Pressure(short) は負が予測。

  目的変数:
    return_{t+k} = mid_pp(t+k) − mid_pp(t)          [pp]   k ∈ {1,2,3,5,10,20,30}

  ★対数リターンを使わない理由: mid は**金利(pp)**であってゼロ近傍・負も取り得る。
    log を取ると定義できない/発散する。金利商品の「リターン」は水準差が自然。

=============================================================================
★時間契約(CLAUDE.md の厳禁事項)
=============================================================================
  - x = Pressure_t は **時刻 t までの情報だけ**で確定する
    (bq_s(t), bq_s(t−1), bq_{o(s)}(t) はいずれも t 時点で観測済み)
  - y = return_{t+k} は **区間 (t, t+k]** のリターン。x の確定時刻 ≤ y の開始時刻 ✓
  - `shift(-k)` は **y にしか使っていない**
  - 標本の選別を結果を見てから行っていない(下記の除外規則はすべて構造的理由)

=============================================================================
除外規則(すべて結果を見る前に決めた構造的理由による)
=============================================================================
  1. `extrapolated`(アーカイブ最終ブロック以降の外挿区間)を除く — 従来報告と同じ
  2. **イベント段差**: 板が片側空・クロスの間は行を出していないので、連続する
     2 行が連続イベントとは限らない。`ev_i` の差が 1 でない箇所を跨ぐ差分・
     窓は除く(「k イベント先」が k イベント先でなくなるため)
  3. **塵の分母**: bq の最小値は 1e-18 YU(= 1 wei)。分母がこれだと Pressure が
     発散し、少数行が結論を支配する。`DUST = 1e-9 YU` 未満の分母を持つ行を除く
  4. **winsorize**: それでも Pressure は裾の重い比なので、主判定は市場内
     [1%, 99%] で winsorize した版。**変数の構成から事前に決めた**処置であって
     結果を見て選んだのではない。生値と Spearman も併記する

  ★実測した病理(結果の読み方に直結するので必ず併記する):
     - 最良数量 bq は 1e-18 〜 1.2e7 の **25 桁**にわたる
     - Pressure(A) は q10〜q90 がすべて厳密に 0。非ゼロ部分は ±1e17 まで伸び、
       生の sd は 1e14 に達する。**winsorize 無しの OLS は数行で決まる**
     - よって「有効標本」は行数 n ではなく **非ゼロ数 n_eff** で見る必要がある

=============================================================================
OLS / GLS
=============================================================================
  重複窓の問題: return_{t+k} は連続する t で k−1 だけ重なる。1 イベントあたりの
  増分が無相関なら、誤差は **MA(k−1)** になり自己相関 ρ_h = 1 − |h|/k を持つ。
  素の OLS 標準誤差はこれを無視するので過小評価になる。よって:

    OLS  : 係数は OLS。標準誤差は **Newey–West**(打ち切りラグ L = k)を主とし、
           比較のため素の標準誤差も出す
    GLS  : **Prais–Winsten の実行可能 GLS**(AR(1) 誤差)。残差から ρ̂ を推定し、
           y*_t = y_t − ρ̂ y_{t−1} 等に変換して OLS を掛け直す

  ★GLS についての注意(結果の解釈に効くので明記する):
    重複が含意する厳密な Ω(MA(k−1) の Toeplitz)は **特異**である。
    k 項移動和のスペクトル密度は ω = 2πm/k でゼロになるため逆行列が存在しない。
    したがって厳密 GLS は使えず、AR(1) 近似を使っている。
    さらに **Pressure_t は先決変数であって厳密外生ではない**。Prais–Winsten 変換は
    x_t と x_{t−1} を混ぜるため、先決変数の下では GLS はむしろ**偏りを持ち得る**。
    この設定で信頼できる推測は OLS + Newey–West のほうである。
    GLS は依頼どおり計算して併記するが、主判定には使わない。

=============================================================================
統計単位(この案件で繰り返し踏んだ罠)
=============================================================================
  全市場をプールした p 値は集塊を無視するので意味がない(前報では
  プール ρ = −0.009 対 市場ごとの中央値 −0.223 と 24 倍違った)。
  主判定は **市場ごとに推定し、符号の一貫性(二項検定)** で行う。

  多重比較: 7 地平 × 3 系列 × 2 推定量 = 42 セル。Bonferroni 閾値 0.05/42 を併記。
  帰無対照: x を市場内で巡回シフトしたプラセボを全セルに置く。

=============================================================================
出力
=============================================================================
  data/pressure_results.parquet   市場 × 地平 × 系列 × 推定量 の推定値
  data/pressure_summary.json      集計
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

HORIZONS = [1, 2, 3, 5, 10, 20, 30]

# 「注文流入量」の重みづけ。**最良気配を最も重くした指数減衰** w_i = exp(−d_i/τ)、
# d_i は最良からの tick 距離。τ→0 が最良 1 レベルのみ、τ→∞ が片側総量。
#   bq    … τ→0 の極限(前報の読み A)
#   ew1〜ew64 … 指数減衰(今回の主題)。τ=16 が板の spread 中央 18 tick に最も近い
#   depth … τ→∞ の極限(前報の読み B)
# 両端を残すのは、τ を動かしたときの梯子として見えるようにするため。
TAGS = ["bq", "ew1", "ew4", "ew16", "ew64", "depth"]
TAU_PRIMARY = "ew16"          # 事前に決めた主系列(spread 中央 18 tick に最も近い τ)
SIGNALS = [f"{t}_{s}" for t in TAGS for s in ("long", "short", "diff")]
DUST = 1e-9          # YU。これ未満の最良数量を分母にしない
WINS = (0.01, 0.99)  # 市場内 winsorize
MIN_N = 500          # これ未満の有効標本の市場は推定しない


# --------------------------------------------------------------------------
# 推定量
# --------------------------------------------------------------------------
def ols_nw(x: np.ndarray, y: np.ndarray, lag: int) -> dict:
    """単回帰 y = a + b x。素の SE と Newey–West SE を返す。"""
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    XtX = X.T @ X
    try:
        XtXi = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return {}
    beta = XtXi @ (X.T @ y)
    u = y - X @ beta
    s2 = u @ u / (n - 2)
    se_plain = np.sqrt(s2 * XtXi[1, 1])

    # Newey–West: S = Σ_h w_h (Γ_h + Γ_h'),  w_h = 1 − h/(L+1)
    # h=0 の項だけを取ったものが White(HC0)。両方返して
    # 「素の SE が外れる原因が不均一分散か系列相関か」を分離できるようにする。
    Xu = X * u[:, None]
    S0 = Xu.T @ Xu
    se_white = np.sqrt(max((XtXi @ S0 @ XtXi)[1, 1], 0.0))
    S = S0.copy()
    for h in range(1, min(lag, n - 1) + 1):
        G = Xu[h:].T @ Xu[:-h]
        S += (1.0 - h / (lag + 1.0)) * (G + G.T)
    V = XtXi @ S @ XtXi
    se_nw = np.sqrt(max(V[1, 1], 0.0))

    ss = ((y - y.mean()) ** 2).sum()
    return {"beta": float(beta[1]), "se_plain": float(se_plain),
            "se_white": float(se_white), "se_nw": float(se_nw),
            "r2": float(1 - (u @ u) / ss) if ss > 0 else np.nan, "n": n}


def gls_pw(x: np.ndarray, y: np.ndarray) -> dict:
    """Prais–Winsten の実行可能 GLS(AR(1) 誤差)。"""
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    try:
        b0 = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {}
    u = y - X @ b0
    den = u[:-1] @ u[:-1]
    rho = float(u[1:] @ u[:-1] / den) if den > 0 else 0.0
    rho = float(np.clip(rho, -0.999, 0.999))

    Xs = X[1:] - rho * X[:-1]
    ys = y[1:] - rho * y[:-1]
    w = np.sqrt(max(1 - rho * rho, 1e-12))
    Xs = np.vstack([w * X[0], Xs])
    ys = np.concatenate([[w * y[0]], ys])

    XtXi = np.linalg.pinv(Xs.T @ Xs)
    beta = XtXi @ (Xs.T @ ys)
    r = ys - Xs @ beta
    s2 = r @ r / max(n - 2, 1)
    return {"beta_gls": float(beta[1]),
            "se_gls": float(np.sqrt(max(s2 * XtXi[1, 1], 0.0))),
            "rho": rho, "n": n}


# --------------------------------------------------------------------------
# 特徴量
# --------------------------------------------------------------------------
def build(path: Path, sfx: str = "") -> tuple[dict, np.ndarray, np.ndarray, int] | None:
    """(信号 dict, mid, 段差なしフラグの累積, 行数) を返す。"""
    mid_id = int(re.search(r"event_book_(?:ex_|band_)?(\d+)", path.name).group(1))
    # sfx="_x" のとき、板の量と mid は除外板の列を読む。
    # ts / ev_i / block / extrapolated は板に依らないので共通。
    need = (["ts", "ev_i", "block", f"mid_pp{sfx}", "extrapolated"]
            + [f"{t}_{s}{sfx}" for t in TAGS for s in ("long", "short")])
    d = pl.read_parquet(path, columns=need).filter(~pl.col("extrapolated")).sort("ts")
    if d.height < MIN_N + max(HORIZONS) + 2:
        return None

    ev = d["ev_i"].to_numpy()
    bl = d[f"bq_long{sfx}"].to_numpy()
    bs = d[f"bq_short{sfx}"].to_numpy()
    mid = d[f"mid_pp{sfx}"].to_numpy()

    # 段差: 前の行と連続イベントか。先頭は不明なので False。
    step1 = np.zeros(len(ev), bool)
    step1[1:] = np.diff(ev) == 1

    def d1(v: np.ndarray) -> np.ndarray:
        o = np.full(len(v), np.nan)
        o[1:] = np.diff(v)
        return o

    # ★分母はどの重みづけでも「反対側の**最良**数量」で固定する(依頼の定義)。
    #   指数減衰にするのは分子(注文流入量)だけ。塵は無効化。
    den_l = np.where(bs >= DUST, bs, np.nan)     # long の分母 = short の最良
    den_s = np.where(bl >= DUST, bl, np.nan)

    sig = {}
    for t in TAGS:
        q_l = d1(d[f"{t}_long{sfx}"].to_numpy()) / den_l
        q_s = d1(d[f"{t}_short{sfx}"].to_numpy()) / den_s
        sig[f"{t}_long"], sig[f"{t}_short"] = q_l, q_s
        sig[f"{t}_diff"] = q_l - q_s
    return ({"market": mid_id, "sig": sig, "mid": mid, "step1": step1,
             "block": d["block"].to_numpy(),
             "n_dust": int((bs < DUST).sum() + (bl < DUST).sum())},
            mid, step1, d.height)


def winsorize(v: np.ndarray, m: np.ndarray) -> np.ndarray:
    ok = m & np.isfinite(v)
    if ok.sum() < 10:
        return v
    lo, hi = np.quantile(v[ok], WINS)
    return np.clip(v, lo, hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--markets", default="", help="カンマ区切り。省略で全市場")
    ap.add_argument("--out", default="pressure_results.parquet")
    ap.add_argument("--tradeable-only", action="store_true",
                    help="窓がブロック境界を跨ぐ標本だけを使う(実装可能性の検査)")
    ap.add_argument("--tags", default="",
                    help="使う重みづけをカンマ区切りで上書きする(既定は bq,ew1,ew4,ew16,ew64,depth)。"
                         "帯の分解を見るときは bq,ib,oob,depth を指定する")
    ap.add_argument("--book", choices=["full", "ex", "band"], default="full",
                    help="ex = 発注量上位 1 口座を除いた板(event_book_ex_*)を使う。"
                         "列は *_x を読む(頑健性検査)")
    a = ap.parse_args()

    global TAGS, SIGNALS
    if a.tags:
        TAGS = [t.strip() for t in a.tags.split(",")]
        SIGNALS = [f"{t}_{s}" for t in TAGS for s in ("long", "short", "diff")]
    pre = {"ex": "event_book_ex_", "band": "event_book_band_"}.get(a.book, "event_book_")
    if a.markets:
        files = [str(DATA / f"{pre}{m}.parquet") for m in a.markets.split(",")]
    elif a.book in ("ex", "band"):
        files = sorted(glob.glob(str(DATA / f"{pre}*.parquet")))
    else:
        files = sorted(f for f in glob.glob(str(DATA / "event_book_*.parquet"))
                       if "event_book_ex_" not in f and "event_book_band_" not in f)
    if a.limit:
        files = files[:a.limit]
    rng = np.random.default_rng(7)
    rows: list[dict] = []
    n_dust_tot = 0
    n_gap_tot = 0
    n_row_tot = 0

    for f in files:
        got = build(Path(f), "_x" if a.book == "ex" else "")
        if got is None:
            continue
        info, mid, step1, nrow = got
        mkt = info["market"]
        n_dust_tot += info["n_dust"]
        n_gap_tot += int((~step1[1:]).sum())
        n_row_tot += nrow

        # 段差の累積和。窓 (t, t+k] に段差が無い ⇔ cum[t+k] − cum[t] == k
        cum = np.concatenate([[0], np.cumsum(step1[1:].astype(np.int64))])

        for k in HORIZONS:
            y = np.full(len(mid), np.nan)
            y[:-k] = mid[k:] - mid[:-k]
            # 窓内に段差なし & x 側も直前と連続
            clean = np.zeros(len(mid), bool)
            clean[:-k] = (cum[k:] - cum[:-k]) == k
            clean &= step1
            if a.tradeable_only:
                # ★窓 (t, t+k] がブロック境界を跨ぐものだけ残す。
                #   跨がない窓は同一ブロック内で完結しており (a) その間に注文を
                #   出せない (b) 1 本のテイカーが複数レベルを掃いた「同じ取引の
                #   続き」を予測と誤認し得る。
                blk = info["block"]
                span = np.zeros(len(mid), bool)
                span[:-k] = blk[k:] > blk[:-k]
                clean &= span

            for name in SIGNALS:
                x0 = info["sig"][name]
                m = clean & np.isfinite(x0) & np.isfinite(y)
                if m.sum() < MIN_N:
                    continue
                xw = winsorize(x0, m)
                xv, yv = xw[m], y[m]
                if np.std(xv) == 0:
                    continue

                o = ols_nw(xv, yv, lag=k)
                g = gls_pw(xv, yv)
                if not o or not g:
                    continue
                sp = stats.spearmanr(xv, yv).statistic
                pe = stats.pearsonr(xv, yv).statistic
                # 生値(winsorize なし)
                xr = x0[m]
                oraw = ols_nw(xr, yv, lag=k) if np.std(xr) > 0 else {}
                # プラセボ: 市場内で x を巡回シフト
                sh = int(rng.integers(len(xv) // 5, 4 * len(xv) // 5))
                op = ols_nw(np.roll(xv, sh), yv, lag=k)

                # ★有効標本: x が厳密に 0 の行は傾きに何も寄与しない。
                #   n ではなく n_eff で検出力を語る必要がある。
                n_eff = int((xv != 0).sum())
                rows.append({
                    "market": mkt, "k": k, "signal": name, "n": int(m.sum()),
                    "n_eff": n_eff, "frac_zero": float((xv == 0).mean()),
                    "beta": o["beta"], "se_plain": o["se_plain"],
                    "se_white": o["se_white"], "se_nw": o["se_nw"],
                    "t_plain": o["beta"] / o["se_plain"] if o["se_plain"] > 0 else np.nan,
                    "t_nw": o["beta"] / o["se_nw"] if o["se_nw"] > 0 else np.nan,
                    "r2": o["r2"],
                    "beta_gls": g["beta_gls"], "se_gls": g["se_gls"],
                    "t_gls": g["beta_gls"] / g["se_gls"] if g["se_gls"] > 0 else np.nan,
                    "rho_ar1": g["rho"],
                    "pearson": pe, "spearman": sp,
                    "beta_raw": oraw.get("beta", np.nan),
                    "t_nw_raw": (oraw["beta"] / oraw["se_nw"]
                                 if oraw and oraw["se_nw"] > 0 else np.nan),
                    "beta_placebo": op.get("beta", np.nan),
                })
        print(f"  {Path(f).name}: {nrow:,} 行", flush=True)

    d = pl.DataFrame(rows)
    d.write_parquet(DATA / a.out)
    print(f"\n市場 {d['market'].n_unique()} / 推定 {d.height:,} 件")
    print(f"段差のある行 {n_gap_tot:,} / {n_row_tot:,} = {n_gap_tot/max(n_row_tot,1):.4%}")
    print(f"塵の最良数量(<{DUST} YU) {n_dust_tot:,} 件")
    print(f"-> {DATA / a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
