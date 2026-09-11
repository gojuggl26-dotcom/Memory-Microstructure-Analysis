"""清算の徹底分析の図(日次・季節性・カスケード・集中・影響)。

    uv run python scripts/plot_liq.py --coin xyz:MU

出力: charts/<tag>_liq_daily.png / _liq_cascade.png / _liq_conc.png / _liq_impact.png

配色は 2 色まで(#3b6fd4 と #c2410c)。palette_check.validate で
`ok=True` を確認済み。3 色目(#0f766e)は彩度下限を割るので使わない。

x が確定する時刻 / y の期間: 図 D-A〜C は記述。図 impact-B のみ
「清算の時刻 τ まで」を条件に「τ より後」を測る(先読みなし)。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _chartstyle import C1, C2, CM, D, plt, save  # noqa: E402

L, S = C2, C1          # ロングの清算 = 橙 / ショートの清算 = 青


def daily(tag, coin):
    day = pl.read_csv(D / f"liqday_{tag}.csv")
    hour = pl.read_csv(D / f"liqhour_{tag}.csv")
    prof = pl.read_csv(D / f"liqprof_{tag}.csv")
    ev = pl.read_parquet(D / f"liqev_{tag}.parquet")
    ed = (ev.group_by("dt", "lside").agg(pl.col("notional").sum().alias("usd"))
          .pivot(values="usd", index="dt", on="lside").fill_null(0).sort("dt"))
    day = day.join(ed, on="dt", how="left").with_columns(
        pl.col("long").fill_null(0), pl.col("short").fill_null(0))
    x = np.arange(day.height)

    fig, ax = plt.subplots(2, 2, figsize=(14.4, 7.6))
    a = ax[0, 0]
    a.bar(x, day["long"] / 1e6, color=L, width=0.85, label="ロングの清算")
    a.bar(x, day["short"] / 1e6, bottom=day["long"] / 1e6, color=S, width=0.85,
          label="ショートの清算")
    a.set_ylabel("清算額(百万 USD)")
    a.set_title("A. 日次の清算額と価格", loc="left")
    a.legend(fontsize=7, frameon=False, loc="upper left")
    a2 = a.twinx()
    a2.plot(x, day["c"], color=CM, lw=1.0)
    a2.set_ylabel("終値(USD)", color=CM)
    a2.grid(False)
    tk = np.arange(0, day.height, 14)
    a.set_xticks(tk)
    dts = day["dt"].to_list()
    a.set_xticklabels([dts[int(i)][5:] for i in tk], fontsize=7)

    a = ax[0, 1]
    d = day.filter(pl.col("liq_usd") > 0)
    a.scatter(d["range_pct"], d["liq_over_vol_bp"], s=22, color=C1, linewidths=0,
              alpha=0.8)
    r = np.corrcoef(np.log(d["range_pct"]), np.log(d["liq_over_vol_bp"]))[0, 1]
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("その日の高値÷安値 − 1(%)")
    a.set_ylabel("清算額 ÷ 出来高(bp)")
    a.set_title(f"B. 値幅が広い日ほど、出来高あたりの清算も多い"
                f"(両対数の相関 {r:.2f}・清算のあった {d.height} 日)", loc="left")

    a = ax[1, 0]
    h = hour["h"].to_numpy()
    a.bar(h + 0.5, hour["liq_per_vol_bp"], color=C1, width=0.86)
    a.set_xlabel("UTC の時刻(棒はその 1 時間)")
    a.set_ylabel("清算額 ÷ 出来高(bp)")
    a.set_xticks(range(0, 25, 3))
    a.set_xlim(0, 24)
    a.set_title("C. 清算が濃い時間帯 — 米株の引けと深夜。寄りは出来高が最大でも清算は薄い",
                loc="left")
    a2 = a.twinx()
    a2.plot(h + 0.5, hour["vol_usd"] / 1e9, color=CM, lw=1.2, marker="o", ms=3)
    a2.set_ylabel("出来高(十億 USD)", color=CM)
    a2.grid(False)
    for xx, lab, ha in ((13.5, "米株 寄り", "right"), (20.0, "米株 引け", "left")):
        a.axvline(xx, color=C2, lw=1.1, ls="--")
        a.text(xx + (0.3 if ha == "left" else -0.3), a.get_ylim()[1] * 0.55, lab,
               fontsize=7.5, color=C2, ha=ha, va="top")

    a = ax[1, 1]
    p = prof.filter(pl.col("liq_usd") > 0)
    a.scatter(p["dwell_h"], p["liq_usd"] / 1e6, s=24, color=C2, linewidths=0, alpha=0.85)
    a.set_xscale("log")
    a.set_yscale("log")
    rr = np.corrcoef(np.log(p["dwell_h"] + 1e-3), np.log(p["liq_usd"]))[0, 1]
    a.set_xlabel("その価格帯に価格が居た時間(時間)")
    a.set_ylabel("その価格帯での清算額(百万 USD)")
    a.set_title(f"D. 「その値段に長く居たから」だけでは説明しきれない"
                f"(両対数の相関 {rr:.2f})", loc="left")
    for r_ in p.sort("liq_per_hour", descending=True).head(3).iter_rows(named=True):
        a.annotate(f"${r_['px_bucket']:.0f}", (r_["dwell_h"], r_["liq_usd"] / 1e6),
                   fontsize=7, xytext=(4, 3), textcoords="offset points", color=C2)
    save(fig, f"{tag}_liq_daily.png",
         f"{coin}: 実際に清算された建玉 — いつ・どの値段で起きたか")


def cascade(tag, coin):
    casc = pl.read_csv(D / f"liqcasc_{tag}.csv")
    ch = pl.read_csv(D / f"liqburst_{tag}.csv")
    slf = pl.read_csv(D / f"liqself_{tag}.csv")
    ev = pl.read_parquet(D / f"liqev_{tag}.parquet").sort("ts")

    fig, ax = plt.subplots(2, 2, figsize=(14.4, 7.6))
    a = ax[0, 0]
    a.bar(casc["gap_s_lo"] + 0.05, casc["n"], width=0.09, color=C1)
    a.set_xlim(0, 12)
    a.set_xlabel("直前の清算ラウンドからの間隔(秒)")
    a.set_ylabel("ラウンド数")
    a.set_title("A. 清算エンジンは約 3 秒の刻みで動く", loc="left")
    for k in (3, 6, 9):
        a.axvline(k, color=C2, lw=0.9, ls="--")
    a.text(3.35, a.get_ylim()[1] * 0.86, "3 秒ごと", fontsize=8, color=C2)

    a = ax[0, 1]
    y = np.arange(slf.height)
    a.barh(y, slf["p_cont"] * 100, color=C1, height=0.6)
    a.axvline(2.25, color=C2, lw=1.2, ls="--")
    a.text(2.9, 3.35, "同じ 1 時間の中で置き直した帰無 2.3%", fontsize=7.5, color=C2,
           va="bottom")
    a.set_yticks(y)
    a.set_yticklabels([f"{c}(n={n:,})" for c, n in zip(slf["class"], slf["n"])],
                      fontsize=8)
    a.invert_yaxis()
    a.set_xlabel("次の刻み(3.5 秒以内)にも清算が続いた割合(%)")
    a.set_title("B. 大きいラウンドほど次を呼ぶ(自己励起)", loc="left")

    a = ax[1, 0]
    v = np.sort(ch["usd"].to_numpy())[::-1]
    a.plot(np.arange(1, v.size + 1), np.cumsum(v) / v.sum() * 100, color=C1, lw=1.4)
    a.set_xscale("log")
    a.set_xlabel("連鎖(上位から数えた本数)")
    a.set_ylabel("累積シェア(%)")
    a.set_title(f"C. {ch.height:,} 本の連鎖のうち上位 50 本で "
                f"{np.cumsum(v)[49]/v.sum()*100:.0f}%", loc="left")
    for k in (10, 50, 200):
        a.axvline(k, color=CM, lw=0.7, ls=":")
        a.text(k, 8, f"{k} 本", fontsize=7, color=CM, ha="center",
               bbox=dict(fc="white", ec="none", pad=0.8))

    a = ax[1, 1]
    t0 = int(ch.sort("usd", descending=True)["t0"][0])
    z = np.load(D / f"_liqmid_{tag}.npz")
    mid, g0 = z["mid"], int(z["t0"])
    s0 = t0 // 10 ** 9
    w0, w1 = s0 - 60, s0 + 120
    xs = np.arange(w0, w1) - s0
    pm = mid[w0 - g0:w1 - g0]
    a.plot(xs, pm, color=CM, lw=1.3, label="mid(1 秒)")
    a.set_xlabel("最大の連鎖の開始からの秒数")
    a.set_ylabel("mid(USD)")
    w = ev.filter((pl.col("ts").dt.epoch("ns") >= w0 * 10 ** 9)
                  & (pl.col("ts").dt.epoch("ns") < w1 * 10 ** 9)).sort("ts")
    tt = (w["ts"].dt.epoch("ns").to_numpy() / 1e9 - s0)
    a2 = a.twinx()
    a2.step(np.r_[xs[0], tt, xs[-1]],
            np.r_[0, np.cumsum(w["notional"].to_numpy()) / 1e6,
                  w["notional"].sum() / 1e6], where="post", color=L, lw=1.5,
            label="清算額の累計")
    a2.set_ylabel("清算額の累計(百万 USD)", color=L)
    a2.grid(False)
    for x_, u_ in zip(tt, w["notional"].to_numpy()):
        a.axvline(x_, color=L, lw=min(3.0, 0.3 + u_ / 3e5), alpha=0.25)
    a.set_title(f"D. 最大の連鎖(${float(ch['usd'][0])/1e6:.2f}M / "
                f"{int(ch['n'][0])} 件 / {float(ch['dur_s'][0]):.0f} 秒)— "
                f"値が落ちて清算が出て、また落ちる", loc="left")
    a.legend(fontsize=7, frameon=False, loc="lower left")
    a2.legend(fontsize=7, frameon=False, loc="upper right")
    save(fig, f"{tag}_liq_cascade.png",
         f"{coin}: 清算の時間構造 — 3 秒の刻み・自己励起・連鎖")


def conc(tag, coin):
    acc = pl.read_csv(D / f"liqacct_{tag}.csv")
    ev = pl.read_parquet(D / f"liqev_{tag}.parquet")
    cp = (ev.group_by("cuid").agg(pl.col("notional").sum().alias("usd"))
          .sort("usd", descending=True))

    fig, ax = plt.subplots(2, 2, figsize=(14.4, 7.6))
    a = ax[0, 0]
    for nm, v, col in (("清算された口座", acc["usd"].to_numpy(), L),
                       ("相手方(取った側)", cp["usd"].to_numpy(), S)):
        v = np.sort(v)
        c = np.cumsum(v) / v.sum() * 100
        a.plot(np.arange(1, v.size + 1) / v.size * 100, c, color=col, lw=1.5,
               label=f"{nm}({v.size:,} 者)")
    a.plot([0, 100], [0, 100], color=CM, lw=0.9, ls="--", label="完全に均等")
    a.set_xlabel("口座(小さい順の累積・%)")
    a.set_ylabel("清算額の累積シェア(%)")
    a.set_title("A. ローレンツ曲線 — どちらも一握りに偏る", loc="left")
    a.legend(fontsize=7, frameon=False, loc="upper left")

    a = ax[0, 1]
    u = np.sort(ev["notional"].to_numpy())
    a.plot(u, 1 - np.arange(u.size) / u.size, color=C1, lw=1.4)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("1 件の清算額(USD)")
    a.set_ylabel("これ以上である割合")
    med = np.median(u)
    a.axvline(med, color=C2, lw=0.9, ls="--")
    a.text(med * 1.15, 0.5, f"中央値 ${med:,.0f}", fontsize=7, color=C2)
    top1 = u[-int(u.size * 0.01):].sum() / u.sum() * 100
    a.set_title(f"B. 1 件あたりの大きさ — 上位 1% で金額の {top1:.0f}%", loc="left")

    a = ax[1, 0]
    n = acc["n"].to_numpy()
    bins = np.array([1, 2, 3, 5, 10, 20, 50, 100, 10 ** 9])
    h = np.array([((n >= bins[i]) & (n < bins[i + 1])).sum() for i in range(bins.size - 1)])
    u2 = np.array([acc.filter((pl.col("n") >= int(bins[i])) & (pl.col("n") < int(bins[i + 1])))
                   ["usd"].sum() for i in range(bins.size - 1)])
    y = np.arange(h.size)
    a.barh(y, h / h.sum() * 100, color=S, height=0.4, label="口座数の割合")
    a.barh(y + 0.42, u2 / u2.sum() * 100, color=L, height=0.4, label="清算額の割合")
    a.set_yticks(y + 0.21)
    a.set_yticklabels(["1 回", "2 回", "3〜4 回", "5〜9 回", "10〜19 回",
                       "20〜49 回", "50〜99 回", "100 回以上"], fontsize=8)
    a.invert_yaxis()
    a.set_xlabel("割合(%)")
    a.set_title(f"C. 清算された回数 — 口座の {h[0]/h.sum()*100:.0f}% は 1 回きりだが、"
                f"金額では {u2[0]/u2.sum()*100:.0f}% しかない", loc="left")
    a.legend(fontsize=7, frameon=False, loc="lower right")

    a = ax[1, 1]
    g = (ev.group_by("cp_dir").agg(pl.col("notional").sum().alias("usd"))
         .sort("usd", descending=True))
    y = np.arange(g.height)
    isopen = np.array([s.startswith("Open") for s in g["cp_dir"]])
    a.barh(y, g["usd"] / 1e6, color=np.where(isopen, S, L), height=0.62)
    a.set_yticks(y)
    a.set_yticklabels(g["cp_dir"], fontsize=8)
    a.invert_yaxis()
    a.set_xlabel("清算ノーショナル(百万 USD)")
    op = float(g.filter(pl.col("cp_dir").str.starts_with("Open"))["usd"].sum()
               / g["usd"].sum() * 100)
    a.set_title(f"D. 相手方が何をしたか — {op:.0f}% は新規建て("
                f"青)なので建玉は消えず移るだけ", loc="left")
    save(fig, f"{tag}_liq_conc.png",
         f"{coin}: 誰が清算され、誰が引き受けたか")


def impact(tag, coin):
    imp = pl.read_csv(D / f"liqimp_{tag}.csv")
    exd = pl.read_csv(D / f"liqexcess_{tag}.csv")
    ev = pl.read_parquet(D / f"liqev_{tag}.parquet")
    vis = pl.read_csv(D / f"liqvisit_{tag}.csv")

    fig, ax = plt.subplots(2, 2, figsize=(14.4, 7.6))
    a = ax[0, 0]
    for nm, col, ls in (("清算(ロング)", L, "-"), ("清算(ショート)", S, "-"),
                        ("対照2 同じ時間枠", CM, "--")):
        s = imp.filter(pl.col("set") == nm)
        xs, ys = [], []
        for lag in sorted(s["lag_s"].unique(), reverse=True):
            v = s.filter((pl.col("lag_s") == lag) & (pl.col("dir") == "bwd"))
            xs.append(-lag)
            ys.append(-float(v["median_bp"][0]))   # 清算時を 0 とした相対価格にする
        xs.append(0)
        ys.append(0.0)
        for lag in sorted(s["lag_s"].unique()):
            v = s.filter((pl.col("lag_s") == lag) & (pl.col("dir") == "fwd"))
            xs.append(lag)
            ys.append(float(v["median_bp"][0]))
        a.plot(np.sign(xs) * np.log10(1 + np.abs(xs)), ys, color=col, lw=1.4, ls=ls,
               marker="o", ms=3, label=nm)
    a.axvline(0, color=CM, lw=0.9)
    a.axhline(0, color=CM, lw=0.6)
    tk = [-3600, -300, -30, 0, 30, 300, 3600]
    a.set_xticks(np.sign(tk) * np.log10(1 + np.abs(tk)))
    a.set_xticklabels(["-1h", "-5m", "-30s", "0", "30s", "5m", "1h"])
    a.set_xlabel("清算の瞬間からの時間")
    a.set_ylabel("清算の瞬間を 0 とした mid(bp・中央値)")
    a.set_title("A. 清算は「動いたあと」に起きる — 直前 5 分で ±80bp 動いている",
                loc="left")
    a.legend(fontsize=7, frameon=False)

    a = ax[1, 0]
    for nm, col in (("ロング", L), ("ショート", S)):
        s = exd.filter(pl.col("side") == nm).sort("lag_s")
        xx = np.log10(s["lag_s"])
        a.plot(xx, s["excess_bp"], color=col, lw=1.5, marker="o", ms=4, label=nm)
        a.fill_between(xx, s["lo"], s["hi"], color=col, alpha=0.16, linewidth=0)
    a.axhline(0, color=CM, lw=0.9)
    for v, lab in ((-1.41, "片道の費用(半スプレッド+テイカー手数料)"), (1.41, None)):
        a.axhline(v, color=CM, lw=0.7, ls=":")
    a.text(np.log10(3400), 1.6, "片道の費用 ±1.41bp", fontsize=7, color=CM,
           ha="right", va="bottom")
    a.set_xticks(np.log10([5, 30, 60, 300, 1800, 3600]))
    a.set_xticklabels(["5s", "30s", "1m", "5m", "30m", "1h"])
    a.set_xlabel("清算からの経過時間")
    a.set_ylabel("超過リターン(bp・中央値)")
    a.set_title("B. 直前 5 分の動きを揃えた超過 — 1 分は押し、1 時間で反転"
                "(帯は日ブロック bootstrap の 95%)", loc="left")
    a.legend(fontsize=7, frameon=False)

    a = ax[0, 1]
    e = ev.filter(pl.col("slip_bp").is_finite())
    q = e.with_columns(dec=((pl.col("notional").rank("ordinal").over("lside")
                             / pl.len().over("lside")) * 10).ceil().clip(1, 10))
    g = (q.group_by("dec", "lside").agg(pl.col("slip_bp").median().alias("m"),
                                        pl.col("notional").median().alias("sz"),
                                        pl.len().alias("n")).sort("dec"))
    for nm, col, lab in (("long", L, "ロングの清算"), ("short", S, "ショートの清算")):
        t = g.filter(pl.col("lside") == nm).sort("dec")
        a.plot(t["dec"], t["m"], color=col, lw=1.5, marker="o", ms=4,
               label=f"{lab}(n={int(t['n'].sum()):,})")
    a.axhline(0, color=CM, lw=0.9)
    a.axhline(1.245 / 2, color=CM, lw=0.7, ls=":")
    a.text(1.1, 1.245 / 2 + 0.15, "板の半スプレッド 0.62bp", fontsize=7, color=CM)
    tl = g.filter(pl.col("lside") == "long").sort("dec")
    a.set_xticks(range(1, 11))
    a.set_xticklabels([str(d) + chr(10) + "$" + format(v, ",.0f")
                       for d, v in zip(tl["dec"], tl["sz"])], fontsize=6.5)
    a.set_xlabel("清算 1 件の大きさ(側ごとの十分位・下段はロング側の中央値)")
    a.set_ylabel("マーク価格からの不利なずれ(bp・中央値)")
    a.set_title("C. 強制執行はマーク価格より 2〜10bp 不利に付く"
                "(ショート側のほうが常に高い)", loc="left")
    a.legend(fontsize=7, frameon=False)

    a = ax[1, 1]
    CLS = [("1 回目", 1, 1), ("2 回目", 2, 2), ("3〜5 回目", 3, 5), ("6 回目以降", 6, 10 ** 9)]
    lab, val = [], []
    for nm, lo_, hi_ in CLS:
        s = vis.filter((pl.col("visit") >= lo_) & (pl.col("visit") <= hi_))
        lab.append(f"{nm}\n({s.height:,} 訪問)")
        val.append(float(s["liq_usd"].sum() / s["vol_usd"].sum() * 1e4))
    a.bar(np.arange(4), val, color=[C2, C1, C1, C1], width=0.64)
    a.set_xticks(np.arange(4))
    a.set_xticklabels(lab, fontsize=8)
    a.set_ylabel("清算額 ÷ その訪問中の出来高(bp)")
    a.set_title("D. 価格帯は「初回の訪問」で最もよく片付く(4.0 倍)", loc="left")
    save(fig, f"{tag}_liq_impact.png",
         f"{coin}: 清算の前後で価格はどう動いたか / 執行の質 / 価格帯の使い減り")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    daily(tag, a.coin)
    cascade(tag, a.coin)
    conc(tag, a.coin)
    impact(tag, a.coin)


if __name__ == "__main__":
    main()
