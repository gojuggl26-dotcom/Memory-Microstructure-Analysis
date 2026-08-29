"""全期間・全約定のテイカー注文を Aggressive Buy / Aggressive Sell に分類する。

【なぜ推定不要か】
  一般のマイクロストラクチャー研究では約定の符号を **tick rule / Lee-Ready で推定**する。
  本案件の L4(node_fills)は 1 取引が **taker 行 1 本 + maker 行 1 本**で構成され、
  `crossed=True` の行がテイカー、その `side` がテイカーの方向である(本日検証済み:
  全 131,800 取引で (crossed=True 1 本, 総 2 行) が 100%、side は左右で完全に相補)。
      side = "B" → **Aggressive Buy**(買い手が板を叩いた)
      side = "A" → **Aggressive Sell**
  したがって**真値**が得られる。

【ついでに測ること】
  真値があるので、**公開データしか無い場合に使う推定則がどれだけ当たるか**を測れる。
    ・tick rule    : 直前の約定価格と比べて上なら買い、下なら売り、同値なら前の符号を継承
    ・quote rule   : 約定価格が mid より上なら買い、下なら売り、mid なら tick rule
                     (Lee-Ready)。mid は**約定時刻より厳密に前**の板から取る
  これが L4 の値段そのものになる。

【出力】
  data/taker_daily.csv    日次集計
  data/taker_summary.json 全体・符号の自己相関・推定則の正答率・層別
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import polars as pl

PIPE = Path("C:/Users/ii562/hl-l4-pipeline")
D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
ACF_LAGS = list(range(1, 21))


def mid_before(dt: str, t_ns: np.ndarray) -> np.ndarray:
    """約定時刻より**厳密に前**の mid。約定後の板を見ないための strict less-than。"""
    fs = glob.glob(str(D / f"microprice/dt={dt}/*.parquet"))
    if not fs:
        return np.full(len(t_ns), np.nan)
    m = (pl.read_parquet(fs[0], columns=["ts", "mid", "is_crossed"])
           .filter(~pl.col("is_crossed")).sort("ts"))
    ts = m["ts"].to_numpy()
    if ts.dtype != np.int64:
        ts = ts.astype("datetime64[ns]").astype(np.int64)
    mid = m["mid"].to_numpy()
    j = np.searchsorted(ts, t_ns, side="left") - 1
    out = np.full(len(t_ns), np.nan)
    ok = j >= 0
    out[ok] = mid[j[ok]]
    return out


def tick_rule(px: np.ndarray) -> np.ndarray:
    """直前の約定価格との比較。同値は前の符号を継承(古典的な tick rule)。"""
    s = np.zeros(len(px))
    d = np.diff(px)
    s[1:] = np.sign(d)
    # 0 の位置は直前の非ゼロ符号を継承
    idx = np.where(s != 0)[0]
    if len(idx) == 0:
        return np.ones(len(px))
    fill = np.searchsorted(idx, np.arange(len(s)), side="right") - 1
    out = np.where(s != 0, s, np.where(fill >= 0, s[idx[np.clip(fill, 0, None)]], 1.0))
    return np.where(out == 0, 1.0, out)


def main() -> None:
    days = sorted(p.name.split("=")[1]
                  for p in (PIPE / "data/fills_v99").glob("dt=*"))
    rows = []
    acf_acc = []
    tick_ok = tick_n = quote_ok = quote_n = 0
    quote_at_mid = 0
    hour_buy = np.zeros(24); hour_all = np.zeros(24)
    strat: dict = {}
    for i, dt in enumerate(days):
        fs = glob.glob(str(PIPE / f"data/fills_v99/dt={dt}/*/part-000.parquet"))
        if not fs:
            continue
        d = pl.concat([pl.read_parquet(x, columns=[
            "ts", "px", "sz", "side", "crossed", "tid", "dir", "user",
            "liquidation", "twapId", "priorityGas", "fee"])
            for x in fs], how="diagonal_relaxed")
        t = d.filter(pl.col("crossed")).unique(subset=["tid"], keep="first")
        if len(t) < 100:
            continue
        ts = t["ts"].to_numpy()
        if ts.dtype != np.int64:
            ts = ts.astype("datetime64[ns]").astype(np.int64)
        o = np.lexsort((t["tid"].to_numpy(), ts))     # 同一 ms 内は tid 順(決定的)
        t = t[o]; ts = ts[o]
        buy = (t["side"].to_numpy() == "B")
        sz = t["sz"].to_numpy(); px = t["px"].to_numpy()
        ntl = px * sz
        sign = np.where(buy, 1.0, -1.0)

        # --- 日次集計 -----------------------------------------------------
        liq = t["liquidation"].fill_null(False).to_numpy()
        tw = t["twapId"].is_not_null().to_numpy()
        pg = t["priorityGas"].is_not_null().to_numpy()
        rows.append({
            "dt": dt, "n_trades": int(len(t)),
            "n_buy": int(buy.sum()), "n_sell": int((~buy).sum()),
            "vol_buy": float(sz[buy].sum()), "vol_sell": float(sz[~buy].sum()),
            "ntl_buy": float(ntl[buy].sum()), "ntl_sell": float(ntl[~buy].sum()),
            "n_liq": int(liq.sum()), "n_twap": int(tw.sum()), "n_prio": int(pg.sum()),
            "buy_share_n": float(buy.mean()),
            "buy_share_ntl": float(ntl[buy].sum() / ntl.sum()),
            "n_users_taker": int(t["user"].n_unique()),
        })
        # --- 符号の自己相関 ----------------------------------------------
        if len(sign) > 5000:
            s = sign - sign.mean()
            v = float((s * s).mean())
            if v > 0:
                acf_acc.append([float((s[:-L] * s[L:]).mean() / v) for L in ACF_LAGS])
        # --- 時間帯 --------------------------------------------------------
        hr = ((ts // 3_600_000_000_000) % 24).astype(int)
        np.add.at(hour_all, hr, 1); np.add.at(hour_buy, hr, buy.astype(float))
        # --- 層別(dir / 清算 / TWAP / 優先手数料)-------------------------
        for key, m in [("dir:" + s_, (t["dir"].to_numpy() == s_))
                       for s_ in t["dir"].unique().to_list() if s_ is not None] + \
                      [("liquidation", liq), ("twap", tw), ("priority_gas", pg),
                       ("regular", ~liq & ~tw & ~pg)]:
            if m.sum() == 0:
                continue
            a = strat.setdefault(key, {"n": 0, "n_buy": 0, "ntl": 0.0, "ntl_buy": 0.0})
            a["n"] += int(m.sum()); a["n_buy"] += int((m & buy).sum())
            a["ntl"] += float(ntl[m].sum()); a["ntl_buy"] += float(ntl[m & buy].sum())
        # --- 推定則の正答率 -------------------------------------------------
        tr = tick_rule(px)
        tick_ok += int((tr == sign).sum()); tick_n += len(sign)
        mb = mid_before(dt, ts)
        ok = np.isfinite(mb)
        qr = np.where(px[ok] > mb[ok], 1.0, np.where(px[ok] < mb[ok], -1.0, 0.0))
        quote_at_mid += int((qr == 0).sum())
        qr = np.where(qr == 0, tr[ok], qr)            # mid ちょうどは tick rule に落とす
        quote_ok += int((qr == sign[ok]).sum()); quote_n += int(ok.sum())
        if (i + 1) % 20 == 0:
            print(f"{i+1}/{len(days)}", flush=True)

    df = pl.DataFrame(rows)
    df.write_csv(D / "taker_daily.csv")
    A = np.vstack(acf_acc) if acf_acc else np.zeros((1, len(ACF_LAGS)))
    summary = {
        "n_days": len(rows),
        "n_trades": int(df["n_trades"].sum()),
        "n_buy": int(df["n_buy"].sum()), "n_sell": int(df["n_sell"].sum()),
        "vol_buy": float(df["vol_buy"].sum()), "vol_sell": float(df["vol_sell"].sum()),
        "ntl_buy": float(df["ntl_buy"].sum()), "ntl_sell": float(df["ntl_sell"].sum()),
        "n_liq": int(df["n_liq"].sum()), "n_twap": int(df["n_twap"].sum()),
        "n_prio": int(df["n_prio"].sum()),
        "acf_lags": ACF_LAGS,
        "acf_mean": A.mean(axis=0).tolist(),
        "acf_days": int(A.shape[0]),
        "tick_rule_accuracy": tick_ok / max(tick_n, 1),
        "quote_rule_accuracy": quote_ok / max(quote_n, 1),
        "quote_at_mid_share": quote_at_mid / max(quote_n, 1),
        "n_classified": tick_n,
        "hour_buy_share": (hour_buy / np.maximum(hour_all, 1)).tolist(),
        "hour_n": hour_all.tolist(),
        "strata": strat,
    }
    (D / "taker_summary.json").write_text(json.dumps(summary, indent=1),
                                          encoding="utf-8")
    print(f"完了: {summary['n_days']} 日 / {summary['n_trades']:,} 取引")


if __name__ == "__main__":
    main()
