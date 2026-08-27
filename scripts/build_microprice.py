"""イベントごとの MicroPrice と midprice を出し、その差から将来リターンの符号確率を作る。

【定義】

    mid_t        = (best_bid + best_ask) / 2
    micro_t      = (best_bid * ask_sz + best_ask * bid_sz) / (bid_sz + ask_sz)

MicroPrice は**反対側の数量**で重みを付ける。買い板が厚い(bid_sz が大きい)ほど
micro は ask 側=上へ寄る。板の厚い側へ価格が動きやすい、という考え方である。

★次の恒等式が厳密に成り立つ(スクリプト内で数値検算する):

    micro_t - mid_t = (I_t - 1/2) * spread_t,   I_t = bid_sz / (bid_sz + ask_sz)

つまり**差は「板の偏り」と「スプレッド幅」の積**であって、2 つの別の量を
混ぜている。スプレッドが広いほど同じ偏りでも差は大きく出る。この点は
解釈上の弱みとして明記し、偏り I で正規化した版も併せて出す。

【x が確定する時刻 / y の期間】
    x = micro_t - mid_t          … 時刻 t の板だけで決まる。t で確定する
    y = mid_{t+k} - mid_t        … 期間は (t, t+k]。t より後にしか判らない
先読みは無い。shift(-k) は y にしか使っていない。

【★同値(y = 0)を必ず分けて報告する】
価格は離散なので、短いホライズンでは「変化なし」が多数を占める。
P(上昇) だけを出すと全セルが 50% を下回って見え、読み手が誤る。
上昇・下落・変化なしの 3 つと、動いた場合に限った P(上昇 | 変化あり) を併記する。

【対照】
- 何もしない場合: そのホライズン・その日区分での**無条件の** P(上昇)。
  各セルはこれと比べる(0.5 と比べない。ドリフトと同値の偏りがあるため)
- 帰無対照: x を日内で巡回シフトして同じ集計をする。偶然なら無条件値に潰れる

【不確かさ】
窓が重なり自己相関があるので二項の信頼区間は使えない。
**日単位のブロックブートストラップ**(99 日を復元抽出)で区間を出す。

    uv run python scripts/build_microprice.py --coin xyz:MU
出力: data/microprice_cells_<coin>.parquet  … 日区分 × 帯 × ホライズンの集計
      data/microprice_daily_<coin>.parquet  … 日 × 日区分 × 帯 × ホライズンの素の計数
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [1, 5, 10, 20, 30, 50, 100]
N_BOOT = 400
SEED = 20260828

# 差(mid に対する bp)の固定帯。対称に取る。
# 実測の分布を見てから決め打ちすると結果を見てからの選択になるので、
# 「対称・等間隔・端は開区間」という機械的な規則で先に決める。
EDGES = [-np.inf, -20.0, -10.0, -5.0, -1.0, 1.0, 5.0, 10.0, 20.0, np.inf]
LABELS = ["< -20", "-20〜-10", "-10〜-5", "-5〜-1", "-1〜+1",
          "+1〜+5", "+5〜+10", "+10〜+20", "> +20"]


def load(coin: str) -> pl.DataFrame:
    tag = coin.replace(":", "_")
    p = ROOT / "data" / f"bbo_{tag}.parquet"
    if not p.exists():
        sys.exit(f"{p} が無い。先に fetch_bbo.py を実行すること")
    d = pl.read_parquet(p).sort("ts")
    n0 = d.height

    # 板が壊れている行を落とす。CLAUDE.md の「is_crossed は必ず除外」に従う。
    d = d.filter(
        (pl.col("best_ask") > pl.col("best_bid"))     # クロス・ロックを除く
        & (pl.col("bid_sz") > 0) & (pl.col("ask_sz") > 0)   # micro が定義できない
        & pl.col("best_bid").is_finite() & pl.col("best_ask").is_finite()
    )
    print(f"[load] {n0:,} 行 -> {d.height:,} 行(クロス・数量 0 を除去 {n0-d.height:,})",
          file=sys.stderr)

    d = d.with_columns(
        mid=(pl.col("best_bid") + pl.col("best_ask")) / 2,
        spread=pl.col("best_ask") - pl.col("best_bid"),
        imb=pl.col("bid_sz") / (pl.col("bid_sz") + pl.col("ask_sz")),
    ).with_columns(
        micro=(pl.col("best_bid") * pl.col("ask_sz") + pl.col("best_ask") * pl.col("bid_sz"))
        / (pl.col("bid_sz") + pl.col("ask_sz")),
    )

    # ★恒等式の検算: micro - mid = (I - 1/2) * spread
    lhs = (d["micro"] - d["mid"]).to_numpy()
    rhs = ((d["imb"] - 0.5) * d["spread"]).to_numpy()
    err = np.nanmax(np.abs(lhs - rhs))
    rel = err / max(np.nanmax(np.abs(lhs)), 1e-12)
    print(f"[検算] micro-mid = (I-1/2)*spread の最大誤差 {err:.3e}(相対 {rel:.3e})",
          file=sys.stderr)
    if rel > 1e-9:
        sys.exit("恒等式が成り立たない。定義かデータを疑うこと")

    return d.with_columns(diff_bp=(pl.col("micro") - pl.col("mid")) / pl.col("mid") * 1e4)


def day_types(days: list[str]) -> dict[str, str]:
    cal = xc.get_calendar("XNYS")
    sess = {x.date().isoformat() for x in cal.sessions_in_range(days[0], days[-1])}
    return {d: ("立会日" if d in sess else "閉場日") for d in days}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    d = load(a.coin)

    days = sorted(d["dt"].unique().to_list())
    dtype = day_types(days)
    print(f"[日] {len(days)} 日 = 立会日 {sum(v=='立会日' for v in dtype.values())}"
          f" / 閉場日 {sum(v=='閉場日' for v in dtype.values())}", file=sys.stderr)

    rng = np.random.default_rng(SEED)
    dt_arr = d["dt"].to_numpy()
    mid = d["mid"].to_numpy()
    x = d["diff_bp"].to_numpy()
    binq = np.digitize(x, EDGES[1:-1])       # 0..len(LABELS)-1

    rows = []
    for day in days:
        m = dt_arr == day
        mi, xi, bi = mid[m], x[m], binq[m]
        n = len(mi)
        if n < 200:
            continue
        # 帰無対照: その日の中で x(と帯)を巡回シフトする。
        # y との対応だけを壊し、x 自身の分布と自己相関は保つ。
        sh = int(rng.integers(n // 5, 4 * n // 5))
        bi_pl = np.roll(bi, sh)
        for k in HORIZONS:
            if n <= k:
                continue
            # ★同一日の中だけで先を見る。日を跨がせない
            #   (跨ぐと閉場日の事象の結果が立会日に実現し、日区分が混ざる)
            y = mi[k:] - mi[:-k]
            b, bp = bi[:-k], bi_pl[:-k]
            up, dn = y > 0, y < 0
            for label_i in range(len(LABELS)):
                s = b == label_i
                tot = int(s.sum())
                if tot == 0:
                    continue
                sp = bp == label_i
                rows.append({
                    "dt": day, "day_type": dtype[day], "bin": LABELS[label_i],
                    "bin_i": label_i, "k": k, "n": tot,
                    "n_up": int(up[s].sum()), "n_dn": int(dn[s].sum()),
                    "n_up_pl": int(up[sp].sum()), "n_pl": int(sp.sum()),
                })
    D = pl.DataFrame(rows)
    D.write_parquet(ROOT / "data" / f"microprice_daily_{tag}.parquet")
    print(f"[集計] 日×帯×ホライズン {D.height:,} 行", file=sys.stderr)

    # ---- 日を単位にブロックブートストラップして区間を出す --------------------
    out = []
    for dtl in ["立会日", "閉場日"]:
        sub = D.filter(pl.col("day_type") == dtl)
        dl = sorted(sub["dt"].unique().to_list())
        idx = {v: i for i, v in enumerate(dl)}
        for k in HORIZONS:
            sk = sub.filter(pl.col("k") == k)
            # 無条件(何もしない場合)= その日区分・そのホライズンの全体
            base_n = sk["n"].sum()
            base_up = sk["n_up"].sum()
            base_dn = sk["n_dn"].sum()
            for label_i, label in enumerate(LABELS):
                c = sk.filter(pl.col("bin_i") == label_i)
                if c.height == 0:
                    continue
                # 日ごとの計数を行列にしてブートストラップ
                nn = np.zeros(len(dl)); uu = np.zeros(len(dl)); dd = np.zeros(len(dl))
                for r in c.iter_rows(named=True):
                    i = idx[r["dt"]]
                    nn[i] += r["n"]; uu[i] += r["n_up"]; dd[i] += r["n_dn"]
                n, nu, nd = nn.sum(), uu.sum(), dd.sum()
                if n < 100:
                    continue
                pick = rng.integers(0, len(dl), size=(N_BOOT, len(dl)))
                bn = nn[pick].sum(1); bu = uu[pick].sum(1)
                bp_ = np.divide(bu, bn, out=np.full(N_BOOT, np.nan), where=bn > 0)
                lo, hi = np.nanpercentile(bp_, [2.5, 97.5])
                mv = nu + nd
                out.append({
                    "day_type": dtl, "bin": label, "bin_i": label_i, "k": k,
                    "n": int(n), "p_up": nu / n, "p_dn": nd / n, "p_flat": (n - mv) / n,
                    "p_up_move": nu / mv if mv > 0 else np.nan,
                    "ci_lo": lo, "ci_hi": hi,
                    "base_up": base_up / base_n,
                    "base_up_move": base_up / (base_up + base_dn) if (base_up + base_dn) else np.nan,
                    "p_up_placebo": c["n_up_pl"].sum() / max(c["n_pl"].sum(), 1),
                })
    C = pl.DataFrame(out)
    C.write_parquet(ROOT / "data" / f"microprice_cells_{tag}.parquet")

    # ---- 画面表示 ------------------------------------------------------------
    for dtl in ["立会日", "閉場日"]:
        s = C.filter(pl.col("day_type") == dtl)
        if s.height == 0:
            continue
        print(f"\n=== {dtl} — P(mid が k イベント後に上昇)[%] ===", file=sys.stderr)
        hdr = f"{'差(bp)':<12}" + "".join(f"{'k='+str(k):>9}" for k in HORIZONS) + f"{'n':>12}"
        print(hdr, file=sys.stderr)
        for label_i, label in enumerate(LABELS):
            r = s.filter(pl.col("bin_i") == label_i).sort("k")
            if r.height == 0:
                continue
            cells = []
            for k in HORIZONS:
                v = r.filter(pl.col("k") == k)
                cells.append(f"{v['p_up'][0]*100:>9.1f}" if v.height else f"{'--':>9}")
            print(f"{label:<12}" + "".join(cells)
                  + f"{r['n'].max():>12,}", file=sys.stderr)
        print(f"{'無条件':<12}" + "".join(
            f"{s.filter(pl.col('k')==k)['base_up'][0]*100:>9.1f}" for k in HORIZONS),
            file=sys.stderr)
        # ★変化なし率は帯ごとの単純平均ではなく件数で加重する。
        #   帯は件数が 118 から 2,600 万まで 5 桁違うので、平均だと極小の帯が
        #   全体と同じ重みを持ってしまい、実態とかけ離れた値になる。
        flat = []
        for k in HORIZONS:
            t = s.filter(pl.col("k") == k)
            flat.append(f"{(t['p_flat']*t['n']).sum()/t['n'].sum()*100:>9.1f}")
        print(f"{'変化なし率':<12}" + "".join(flat), file=sys.stderr)
        mv = []
        for k in HORIZONS:
            t = s.filter(pl.col("k") == k)
            mv.append(f"{t['base_up_move'][0]*100:>9.1f}")
        print(f"{'動いた中で上昇':<11}" + "".join(mv), file=sys.stderr)
    print(f"\n-> data/microprice_cells_{tag}.parquet", file=sys.stderr)


if __name__ == "__main__":
    main()
