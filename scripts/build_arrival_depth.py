"""板の深さごとの注文到着率(1 秒あたり)を、6 つの時間帯について買い・売り別に出す。

【到着とは何か】
L1 の `status == "open"` の行を「到着」とする。**その注文が実際に板に載った**瞬間で、
価格 `px` と側 `side` が判る。即座に約定して板に載らなかった注文や、
拒否された注文(`badAloPxRejected` 等)は板に並ばないので含めない。

【深さ】
到着時点の mid からの距離(bp)。**mid はその到着より厳密に前の BBO** を使う
(`join_asof(strategy="backward", allow_exact_matches=False)`)。同時刻の BBO は
その到着自身を反映している可能性があり、距離が 0 側に寄ってしまうため。

    買い(B): depth = (mid − px) / mid × 10^4
    売り(A): depth = (px − mid) / mid × 10^4

正が板の内側(mid から遠い)、負は mid を跨ぐ攻撃的な価格。
帯はパイプラインの BP_BUCKETS に合わせる(0-1,1-2,2-5,5-10,10-25,25-50,50-100,100+)。

【時間帯】
標本期間は全日が米国東部夏時間(ET = UTC−4)。立会は 09:30–16:00 ET = 13:30–20:00 UTC。
「昼 12 時」「夜 12 時」は **ET** と読む。UTC と読むと開場前・閉場後の帯と重なって
別々に測る意味が薄れるため。UTC と読んだ場合の 2 帯も併せて出しておく。

    開場前 1 時間     12:30–13:30 UTC  (08:30–09:30 ET)   立会日
    開場後 1 時間     13:30–14:30 UTC  (09:30–10:30 ET)   立会日
    昼 12 時 (ET)     16:00–17:00 UTC  (12:00–13:00 ET)   立会日
    閉場前 1 時間     19:00–20:00 UTC  (15:00–16:00 ET)   立会日
    夜 12 時 (ET)     04:00–05:00 UTC  (00:00–01:00 ET)   立会日
    閉場日の昼 12 時  16:00–17:00 UTC  (12:00–13:00 ET)   閉場日
    (参考) 昼 12 時 UTC  12:00–13:00 UTC 立会日
    (参考) 夜 12 時 UTC  00:00–01:00 UTC 立会日

【費用】
L1 は Glacier IR にあるので取得料がかかる。必要 5 列 + 時刻で行グループを絞るため、
1 日あたり約 24 MB(全体 541 MB の 4%)しか読まない。

x が確定する時刻 / y の期間: 該当なし(実測量の集計であって予測ではない)。

    uv run python scripts/build_arrival_depth.py --coin xyz:MU
出力: data/arrival_depth_<coin>.parquet   … 日 × 時間帯 × 側 × 深さ帯の件数と数量
      data/arrival_depth_<coin>.csv       … 時間帯 × 側 × 深さ帯の要約
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import exchange_calendars as xc
import numpy as np
import polars as pl
import pyarrow.fs as pafs
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
COLS = ["ts", "px", "side", "status", "orig_sz"]

# (名前, 開始 UTC 時, 日区分)。長さはすべて 1 時間
WINDOWS = [
    ("開場前 1h",        12, "立会日"),
    ("開場後 1h",        13, "立会日"),
    ("昼 12 時 ET",      16, "立会日"),
    ("閉場前 1h",        19, "立会日"),
    ("夜 12 時 ET",       4, "立会日"),
    ("閉場日 昼 12 時 ET", 16, "閉場日"),
    ("(参考) 昼 12 時 UTC", 12, "立会日"),   # 開場前 1h と同じ時間帯
    ("(参考) 夜 12 時 UTC",  0, "立会日"),
]
# 「開場前 1h」は 12:30 開始なので特別扱いする
HALF_HOUR_START = {"開場前 1h": 30, "開場後 1h": 30}

BP_EDGES = [0.0, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]
# 深さが負 = mid より内側(その側の気配を mid の向こう側まで改善した注文)。
# 反対側の気配は跨げない(跨げば即約定して板に載らない)ので「跨ぐ」ではない。
BP_LABELS = ["mid より内", "0–1", "1–2", "2–5", "5–10", "10–25", "25–50", "50–100", "100+"]
NB = len(BP_LABELS)


def win_bounds(day: dt.date, name: str, hour: int) -> tuple[dt.datetime, dt.datetime]:
    m = HALF_HOUR_START.get(name, 0)
    s = dt.datetime.combine(day, dt.time(hour, m))
    return s, s + dt.timedelta(hours=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--days", type=int, default=0, help="先頭 N 日だけ(検証用)")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    bucket = os.environ.get("WORK_BUCKET") or sys.exit("WORK_BUCKET 未設定")

    # --- BBO(手元)。mid の供給元 ---------------------------------------------
    bbo = (
        pl.read_parquet(ROOT / "data" / f"bbo_{tag}.parquet")
        .filter((pl.col("best_ask") > pl.col("best_bid"))
                & (pl.col("bid_sz") > 0) & (pl.col("ask_sz") > 0))
        .with_columns(mid=(pl.col("best_bid") + pl.col("best_ask")) / 2)
        # bbo の ts は生のナノ秒(Int64)、L1 は Datetime(ns)。結合のため型を揃える
        .with_columns(ts=pl.col("ts").cast(pl.Datetime("ns")))
        .select("ts", "mid", "dt")
        .sort("ts")
    )
    bbo_days = set(bbo["dt"].unique().to_list())
    print(f"[BBO] {bbo.height:,} 行 / {len(bbo_days)} 日", file=sys.stderr)

    inv = pl.read_parquet(ROOT / "data" / "s3_inventory.parquet")
    keys = (
        inv.filter((pl.col("coin") == a.coin) & (pl.col("layer") == "l1")
                   & (pl.col("leaf") == "part-000.parquet"))
        .sort("dt").select("dt", "key").rows()
    )
    # BBO の無い日は mid が作れないので落とす(L1 は 99 日、BBO は 98 日)
    skipped = [d for d, _ in keys if d not in bbo_days]
    keys = [(d, k) for d, k in keys if d in bbo_days]
    if skipped:
        print(f"[skip] BBO が無い日を除外: {skipped}", file=sys.stderr)
    if a.days:
        keys = keys[:a.days]

    cal = xc.get_calendar("XNYS")
    days_all = [d for d, _ in keys]
    sess = {x.date().isoformat() for x in cal.sessions_in_range(days_all[0], days_all[-1])}
    dtype = {d: ("立会日" if d in sess else "閉場日") for d in days_all}
    print(f"[日] {len(keys)} 日 = 立会日 {sum(v == '立会日' for v in dtype.values())}"
          f" / 閉場日 {sum(v == '閉場日' for v in dtype.values())}", file=sys.stderr)

    sess_b = boto3.Session(profile_name=os.environ.get("AWS_PROFILE", "hl-artemis-ro"))
    cr = sess_b.get_credentials().get_frozen_credentials()
    fs = pafs.S3FileSystem(access_key=cr.access_key, secret_key=cr.secret_key,
                           session_token=cr.token, region="us-east-1")

    read_bytes = [0]
    # ★日ごとにキャッシュする。Glacier IR は取得のたびに課金されるので、
    #   途中で止まっても、済んだ日を読み直して二重に払わないようにする。
    cache = ROOT / "data" / "_arrival_cache"
    cache.mkdir(exist_ok=True)

    def one(item):
        day_s, key = item
        cf = cache / f"{tag}_{day_s}.parquet"
        if cf.exists():
            return [] if cf.stat().st_size == 0 else pl.read_parquet(cf).to_dicts()
        day = dt.date.fromisoformat(day_s)
        want = [(nm, h) for nm, h, dl in WINDOWS if dl == dtype[day_s]]
        if not want:
            return []
        spans = [(nm, *win_bounds(day, nm, h)) for nm, h in want]
        lo = min(s for _, s, _ in spans)
        hi = max(e for _, _, e in spans)

        f = fs.open_input_file(f"{bucket}/{key}")
        pf = pq.ParquetFile(f)
        m = pf.metadata
        ti = pf.schema_arrow.names.index("ts")
        sel, nb = [], 0
        for i in range(m.num_row_groups):
            rg = m.row_group(i)
            st = rg.column(ti).statistics
            if st is None or (st.max >= lo and st.min <= hi):
                sel.append(i)
                for j in range(rg.num_columns):
                    c = rg.column(j)
                    if c.path_in_schema in COLS:
                        nb += c.total_compressed_size
        if not sel:
            return []
        read_bytes[0] += nb
        d = pl.from_arrow(pf.read_row_groups(sel, columns=COLS))
        d = d.filter(pl.col("status") == "open").select("ts", "px", "side", "orig_sz").sort("ts")
        if d.height == 0:
            cf.touch()          # 読んだ事実を残す(再取得を防ぐ)
            return []

        b = bbo.filter(pl.col("dt") == day_s).select("ts", "mid")
        # ★到着より厳密に前の BBO を使う。同時刻はその到着自身を含みうる
        d = d.join_asof(b, on="ts", strategy="backward", allow_exact_matches=False)
        d = d.drop_nulls("mid").with_columns(
            depth=pl.when(pl.col("side") == "B")
            .then((pl.col("mid") - pl.col("px")) / pl.col("mid") * 1e4)
            .otherwise((pl.col("px") - pl.col("mid")) / pl.col("mid") * 1e4)
        )
        out = []
        for nm, s, e in spans:
            w = d.filter((pl.col("ts") >= s) & (pl.col("ts") < e))
            if w.height == 0:
                continue
            band = np.digitize(w["depth"].to_numpy(), BP_EDGES)  # 0 = mid を跨ぐ
            side = (w["side"].to_numpy() == "B")
            sz = w["orig_sz"].to_numpy()
            for is_bid, lab in ((True, "買い(bid)"), (False, "売り(ask)")):
                msk = side == is_bid
                cnt = np.bincount(band[msk], minlength=NB)
                ssz = np.bincount(band[msk], weights=sz[msk], minlength=NB)
                for i in range(NB):
                    if cnt[i] == 0:
                        continue
                    out.append({"dt": day_s, "day_type": dtype[day_s], "window": nm,
                                "side": lab, "band_i": i, "band": BP_LABELS[i],
                                "n": int(cnt[i]), "sz": float(ssz[i])})
        # S3 を読んだ日は必ずキャッシュに残す(空でも再取得しないため)
        pl.DataFrame(out, schema={
            "dt": pl.String, "day_type": pl.String, "window": pl.String,
            "side": pl.String, "band_i": pl.Int64, "band": pl.String,
            "n": pl.Int64, "sz": pl.Float64,
        }).write_parquet(cf)
        return out

    rows = []
    with ThreadPoolExecutor(a.threads) as ex:
        for i, r in enumerate(ex.map(one, keys), 1):
            rows.extend(r)
            if i % 10 == 0 or i == len(keys):
                print(f"  {i}/{len(keys)}  読み {read_bytes[0]/1e9:.2f} GB", file=sys.stderr, flush=True)

    D = pl.DataFrame(rows)
    D.write_parquet(ROOT / "data" / f"arrival_depth_{tag}.parquet")

    # --- 1 秒あたりに直して要約 ------------------------------------------------
    # 日ごとに率を出してから日をまたいで平均する(1 日 1 票。活発な日に引きずられない)
    #
    # ★到着が 1 件も無かった (窓, 側, 帯) の日は D に行が立たない。そのまま
    #   group_by すると分母が「到着があった日数」になり、まばらな帯の率を
    #   過大に出す(平均も中央値も分位も 0 の日を数えなくなる)。窓ごとの
    #   観測日を軸に 0 を埋めてから平均する。実測では 98 日で欠けるセルは
    #   無かったが、静かな銘柄や短い窓では必ず効くので構造として直しておく。
    obs = D.select("window", "dt").unique()
    full = (
        obs.join(pl.DataFrame({"side": ["買い(bid)", "売り(ask)"]}), how="cross")
        .join(pl.DataFrame({"band_i": list(range(NB)), "band": BP_LABELS}), how="cross")
        .join(D.select("window", "dt", "side", "band_i", "n", "sz"),
              on=["window", "dt", "side", "band_i"], how="left")
        .with_columns(pl.col("n").fill_null(0), pl.col("sz").fill_null(0.0))
    )
    n_empty = int((full["n"] == 0).sum())
    if n_empty:
        print(f"[0 埋め] 到着ゼロのセル {n_empty:,} / {full.height:,}", file=sys.stderr)
    per_day = full.with_columns(rate=pl.col("n") / 3600.0, rate_sz=pl.col("sz") / 3600.0)
    n_days = (per_day.group_by("window").agg(d=pl.col("dt").n_unique())
              .rename({"d": "n_days"}))
    S = (
        per_day.group_by("window", "side", "band", "band_i")
        .agg(rate_mean=pl.col("rate").sum() / pl.col("dt").n_unique(),
             rate_med=pl.col("rate").median(),
             rate_p10=pl.col("rate").quantile(0.10),
             rate_p90=pl.col("rate").quantile(0.90),
             sz_mean=pl.col("rate_sz").sum() / pl.col("dt").n_unique(),
             n_days=pl.col("dt").n_unique(),
             n_total=pl.col("n").sum())
        .sort("window", "side", "band_i")
    )
    S.write_csv(ROOT / "data" / f"arrival_depth_{tag}.csv")

    order = [w[0] for w in WINDOWS]
    print(f"\n=== 深さ帯ごとの注文到着率(件/秒)。日ごとに率を出して日平均 ===", file=sys.stderr)
    for nm in order:
        s = S.filter(pl.col("window") == nm)
        if s.height == 0:
            continue
        nd = int(s["n_days"].max())
        print(f"\n--- {nm}({nd} 日)---", file=sys.stderr)
        print(f"{'側':<12}" + "".join(f"{b:>10}" for b in BP_LABELS) + f"{'合計':>10}",
              file=sys.stderr)
        for side in ("買い(bid)", "売り(ask)"):
            t = s.filter(pl.col("side") == side)
            cells = []
            for i in range(NB):
                r = t.filter(pl.col("band_i") == i)
                cells.append(f"{r['rate_mean'][0]:>10.2f}" if r.height else f"{'--':>10}")
            print(f"{side:<12}" + "".join(cells) + f"{t['rate_mean'].sum():>10.2f}",
                  file=sys.stderr)
    print(f"\n[読み取り] {read_bytes[0]/1e9:.2f} GB(Glacier IR 取得 + 転送の概算 "
          f"${read_bytes[0]/1e9*0.12:.2f})", file=sys.stderr)
    print(f"-> data/arrival_depth_{tag}.parquet / .csv", file=sys.stderr)


if __name__ == "__main__":
    main()
