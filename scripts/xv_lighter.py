r"""会場間リードラグ — Lighter と Binance(同一ホスト記録・同一クロック)。

    uv run python scripts/xv_lighter.py --sym MU
    uv run python scripts/xv_lighter.py --sym MU --clock exch   # 対照

出力:
    data/xvl_ccfday_<sym>_<clock>.parquet … 日 × 時間帯 × ラグ の相関
    data/xvl_daily_<sym>_<clock>.csv      … 日次の素性
    data/xvl_dev_<sym>_<clock>.csv        … 乖離と半減期

=============================================================================
★この解析が Hyperliquid 版より強い理由 — 時計が 1 つで済む
=============================================================================
`xyz:MU` × Binance では**両会場のクロック差が観測できず**、ピーク位置の
点推定を主張できなかった。ここではその制約が無い:

  Lighter  … `recv_ns`  = このホストが受信した時刻
  Binance  … `local_ns` = このホストが受信した時刻

**両方とも同じマシンの同じ時計**である(記録プロセスは PID 21808 / 42844 で
同時刻に起動し、同じディスクへ書いている)。したがって

  **この時計の上で測った先行は、そのホストに居るトレーダーが実際に見る先行**

であり、補正を要しない。これは「取引所の時計での先行」とは別物で、
**戦略に使えるのは後者ではなく前者**である。

★対照として `--clock exch`(取引所の打刻)も回せる。Binance の記録は
  `offset_ms = −368 〜 −377ms`(rtt 88ms)を実測しており、**取引所時刻を
  そのまま比べると数百 ms ずれる**。どれだけ結論が変わるかを見るため。

=============================================================================
★gzip の切断トラップ(既知・再発)
=============================================================================
記録プロセスが kill された gz に追記されると、**標準リーダは切断点で止まり
以降を黙って捨てる**。実測(MU の 1 日)で標準 537,766 行に対し
耐性リーダは **572,110 行**(+6.0%)。メンバー単位で走査して読む。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import zlib
from pathlib import Path

import numpy as np
import polars as pl

LGT = Path("E:/Lighter　データ/data/raw")
BNC = Path("E:/Binance-perp-data/data/raw")
DATA = Path("C:/Users/ii562/Downloads/Memory/data")
BP = 1e4
NS = 1_000_000_000
SESS = [("00-08 夜", 0, 8), ("08-13.5 プレ", 8, 13.5),
        ("13.5-20 現物", 13.5, 20), ("20-24 アフター", 20, 24)]


def read_gz_tolerant(path):
    """メンバー単位で読み、切断されたメンバーからも読めた分を回収する。

    返り値 (bytes, 壊れたメンバー数)。切れた行は捨てる。
    """
    raw = open(path, "rb").read()
    out, i, nbad = [], 0, 0
    while True:
        j = raw.find(b"\x1f\x8b", i)
        if j < 0:
            break
        d = zlib.decompressobj(31)
        buf, k, err = [], j, False
        while k < len(raw):
            try:
                buf.append(d.decompress(raw[k:k + (1 << 20)]))
            except Exception:
                err = True
                break
            k += (1 << 20)
            if d.eof:
                break
        b = b"".join(buf)
        # ★例外が出なくても、データが尽きて eof に達しなければ切断である
        if err or not d.eof:
            nbad += 1
            b = b[:b.rfind(b"\n") + 1]
            out.append(b)
            i = j + 2
        else:
            out.append(b)
            rest = getattr(d, "unused_data", b"")
            if not rest:
                break
            i = len(raw) - len(rest)
    return b"".join(out), nbad


CACHE = DATA / "cache"


def cached(kind, sym, day, fn):
    """解析済みの断面を parquet に貯める(版管理外)。3 構成を回すため。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"xvl_{kind}_{sym}_{day}.parquet"
    if f.exists():
        D = pl.read_parquet(f)
        return (D["t_local"].to_numpy(), D["t_exch"].to_numpy(),
                D["bid"].to_numpy(), D["ask"].to_numpy(),
                int(D["gz_bad"][0]))
    r = fn()
    if r is None:
        return None
    # ★原子的に書く。途中で落ちた 0 バイトを次回読んで落ちるのを防ぐ
    tmp = f.with_suffix(".tmp")
    pl.DataFrame({"t_local": r[0], "t_exch": r[1], "bid": r[2], "ask": r[3],
                  "gz_bad": np.full(r[0].size, r[4], np.int32)}
                 ).write_parquet(tmp)
    tmp.replace(f)
    return r


def load_lighter(sym, day):
    """Lighter の ticker から (recv_ns, exch_ns, bid, ask)。"""
    fs = sorted(glob.glob(str(LGT / "ws/ticker" / sym / f"dt={day}" / "*.gz")))
    if not fs:
        return None
    rows, nbad = [], 0
    for f in fs:
        b, nb = read_gz_tolerant(f)
        nbad += nb
        rows.append(b)
    buf = b"".join(rows)
    if not buf:
        return None
    rv, ex, bd, ak = [], [], [], []
    for line in buf.split(b"\n"):
        if not line:
            continue
        try:
            o = json.loads(line)
            t = o["m"]["ticker"]
            rv.append(o["recv_ns"])
            ex.append(o["m"]["last_updated_at"] * 1000)   # us -> ns
            bd.append(float(t["b"]["price"]))
            ak.append(float(t["a"]["price"]))
        except Exception:
            continue
    if not rv:
        return None
    r = np.array(rv, np.int64)
    o = np.argsort(r, kind="stable")
    return (r[o], np.array(ex, np.int64)[o],
            np.array(bd)[o], np.array(ak)[o], nbad)


def load_binance_trade(sym, day):
    """Binance の trade から (local_ns, exch_ns, px, px)。

    ★Hyperliquid 版と条件を揃えるための対照。あちらは Binance Vision の
      aggTrades(約定値)しか使えなかったので、気配ではなく約定値で測ると
      相関がどれだけ落ちるかを、**同じ期間・同じ相手**で切り分ける。
    """
    p = BNC / f"dt={day}" / f"symbol={sym}USDT" / "stream=trade"
    fs = [f for f in sorted(glob.glob(str(p / "*.parquet")))
          if os.path.getsize(f) > 8]
    if not fs:
        return None
    parts = []
    for f in fs:
        try:
            parts.append(pl.read_parquet(
                f, columns=["local_ns", "event_ms", "payload"]))
        except Exception:
            continue
    if not parts:
        return None
    D = (pl.concat(parts).with_columns(
        pl.col("payload").str.json_path_match("$.data.p")
        .cast(pl.Float64).alias("px"))
        .drop("payload").drop_nulls().sort("local_ns"))
    if D.height == 0:
        return None
    px = D["px"].to_numpy()
    return (D["local_ns"].to_numpy(),
            D["event_ms"].to_numpy().astype(np.int64) * 1_000_000,
            px, px, 0)


def load_binance(sym, day):
    """Binance の bookTicker から (local_ns, exch_ns, bid, ask)。"""
    p = BNC / f"dt={day}" / f"symbol={sym}USDT" / "stream=bookTicker"
    fs = sorted(glob.glob(str(p / "*.parquet")))
    # ★記録プロセスが動いているので、書きかけ・0 バイトの part が混じる。
    #   除かないと polars が「footer が無い」で落ちる(実際に落ちた)。
    fs = [f for f in fs if os.path.getsize(f) > 8]
    if not fs:
        return None
    parts = []
    for f in fs:
        try:
            parts.append(pl.read_parquet(
                f, columns=["local_ns", "event_ms", "payload"]))
        except Exception:
            continue          # 書きかけの 1 本で 1 日を落とさない
    if not parts:
        return None
    D = pl.concat(parts)
    D = D.with_columns(
        pl.col("payload").str.json_path_match("$.data.b")
        .cast(pl.Float64).alias("bid"),
        pl.col("payload").str.json_path_match("$.data.a")
        .cast(pl.Float64).alias("ask"),
    ).drop("payload").drop_nulls().sort("local_ns")
    if D.height == 0:
        return None
    return (D["local_ns"].to_numpy(),
            D["event_ms"].to_numpy().astype(np.int64) * 1_000_000,
            D["bid"].to_numpy(), D["ask"].to_numpy(), 0)


def locf(ts, v, q):
    i = np.searchsorted(ts, q, side="right") - 1
    ok = i >= 0
    i = np.clip(i, 0, len(ts) - 1)
    return np.where(ok, v[i], np.nan), np.where(ok, q - ts[i], np.inf)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sym", default="MU")
    ap.add_argument("--clock", choices=["local", "exch"], default="local")
    ap.add_argument("--cell", type=int, default=100)
    ap.add_argument("--lags", type=int, default=20)
    ap.add_argument("--fresh", type=float, default=2.0)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--bsrc", choices=["quote", "trade"], default="quote",
                    help="Binance 側を気配にするか約定値にするか(対照)")
    ap.add_argument("--days", type=int, default=0)
    a = ap.parse_args()
    cell = a.cell * 1_000_000
    nlag = a.lags
    sfx = (f"_{a.clock}_c{a.cell}" + ("_strict" if a.strict else "")
           + ("_btrade" if a.bsrc == "trade" else ""))

    ld = {d.split("=")[1] for d in os.listdir(LGT / "ws/ticker" / a.sym)}
    bd = {d.split("=")[1] for d in os.listdir(BNC) if d.startswith("dt=")}
    days = sorted(ld & bd)
    if a.days:
        days = days[:a.days]
    print(f"Lighter {a.sym} × Binance {a.sym}USDT / 時計 = "
          f"{'ローカル受信' if a.clock=='local' else '取引所打刻'} / "
          f"{len(days)} 日 / 格子 {a.cell}ms", flush=True)

    per_day, daily, devrows = [], [], []
    ccf_n = np.zeros((len(SESS) + 1, 2 * nlag + 1))
    ccf_s = np.zeros_like(ccf_n)
    for day in days:
        L = cached("lgt", a.sym, day, lambda: load_lighter(a.sym, day))
        bfn = load_binance if a.bsrc == "quote" else load_binance_trade
        B = cached("bnc" if a.bsrc == "quote" else "bnctr", a.sym, day,
                   lambda: bfn(a.sym, day))
        if L is None or B is None or L[0].size < 5000 or B[0].size < 5000:
            daily.append({"dt": day, "ok": False,
                          "n_lgt": 0 if L is None else int(L[0].size),
                          "n_bnc": 0 if B is None else int(B[0].size)})
            continue
        ci = 0 if a.clock == "local" else 1
        tl, ml = L[ci], 0.5 * (L[2] + L[3])
        tb, mb = B[ci], 0.5 * (B[2] + B[3])
        okl = np.isfinite(ml) & (L[3] > L[2]) & (L[2] > 0)
        okb = (np.isfinite(mb) & (B[2] > 0)
               & ((B[3] > B[2]) | (a.bsrc == "trade")))
        tl, ml, tb, mb = tl[okl], ml[okl], tb[okb], mb[okb]
        o = np.argsort(tl, kind="stable")
        tl, ml = tl[o], ml[o]
        o = np.argsort(tb, kind="stable")
        tb, mb = tb[o], mb[o]

        d0 = (tl[0] // (86400 * NS)) * 86400 * NS
        grid = np.arange(d0, d0 + 86400 * NS, cell, dtype=np.int64)
        vl, al = locf(tl, ml, grid)
        vb, ab = locf(tb, mb, grid)
        good = (np.isfinite(vl) & np.isfinite(vb)
                & (al <= a.fresh * NS) & (ab <= a.fresh * NS))
        if a.strict:
            good &= (al < cell) & (ab < cell)
        if good.sum() < 5000:
            daily.append({"dt": day, "ok": False, "n_lgt": int(tl.size),
                          "n_bnc": int(tb.size)})
            continue
        ll, lb = np.log(vl), np.log(vb)
        w = max(10, int(30 * 60 * 1000 / a.cell))
        raw = (lb - ll) * BP
        base = (pl.Series(np.where(good, raw, np.nan))
                .rolling_median(w, min_samples=max(10, w // 20)).to_numpy())
        dev = raw - base
        rl = np.concatenate([[np.nan], np.diff(ll)]) * BP
        rb = np.concatenate([[np.nan], np.diff(lb)]) * BP
        hour = (grid - d0) / (3600 * NS)
        sid = np.full(grid.size, -1, np.int8)
        for i, (_, x, y) in enumerate(SESS):
            sid[(hour >= x) & (hour < y)] = i

        # corr(rb[t], rl[t+k]) — k>0 なら Binance が先、Lighter が後
        for k in range(-nlag, nlag + 1):
            x = rb[max(0, -k): rb.size - max(0, k)]
            y = rl[max(0, k): rl.size - max(0, -k)]
            g = (good[max(0, -k): good.size - max(0, k)]
                 & good[max(0, k): good.size - max(0, -k)]
                 & np.isfinite(x) & np.isfinite(y))
            if g.sum() < 100:
                continue
            c = np.corrcoef(x[g], y[g])[0, 1]
            if not np.isfinite(c):
                continue
            j = k + nlag
            ccf_n[0, j] += 1
            ccf_s[0, j] += c
            per_day.append({"dt": day, "group": "全体",
                            "lag_ms": int(k * a.cell), "corr": float(c),
                            "n": int(g.sum())})
            ss = sid[max(0, -k): sid.size - max(0, k)]
            for i in range(len(SESS)):
                gg = g & (ss == i)
                if gg.sum() < 100:
                    continue
                cc = np.corrcoef(x[gg], y[gg])[0, 1]
                if np.isfinite(cc):
                    ccf_n[i + 1, j] += 1
                    ccf_s[i + 1, j] += cc
                    per_day.append({"dt": day, "group": SESS[i][0],
                                    "lag_ms": int(k * a.cell),
                                    "corr": float(cc), "n": int(gg.sum())})

        dd = np.where(good, dev, np.nan)
        x0, x1 = dd[:-1], dd[1:]
        m = np.isfinite(x0) & np.isfinite(x1)
        phi = (float((x0[m] * x1[m]).sum() / max((x0[m] ** 2).sum(), 1e-12))
               if m.sum() > 100 else np.nan)
        devrows.append({
            "dt": day, "phi": phi,
            "half_life_ms": (np.log(.5) / np.log(phi) * a.cell
                             if np.isfinite(phi) and 0 < phi < 1 else np.nan),
            "dev_abs_p50": float(np.nanmedian(np.abs(dd))),
            "dev_abs_p90": float(np.nanquantile(np.abs(dd), .90)),
            "basis_p50": float(np.nanmedian(np.where(good, raw, np.nan))),
        })
        daily.append({
            "dt": day, "ok": True, "n_lgt": int(tl.size), "n_bnc": int(tb.size),
            "gz_bad": int(L[4]),
            "usable_pct": float(100 * good.mean()),
            "lgt_fresh_pct": float(100 * np.mean(al <= a.fresh * NS)),
            "bnc_fresh_pct": float(100 * np.mean(ab <= a.fresh * NS)),
            "lgt_move_pct": float(100 * np.mean(np.abs(rl[good]) > 1e-9)),
            "bnc_move_pct": float(100 * np.mean(np.abs(rb[good]) > 1e-9)),
            # 取引所打刻から手元に届くまでの実測(ローカル時計で見た差)
            "lgt_lat_ms_p50": float(np.median(L[0] - L[1]) / 1e6),
            "bnc_lat_ms_p50": float(np.median(B[0] - B[1]) / 1e6),
        })
        print(f"  {day} Lighter {tl.size:>8,} / Binance {tb.size:>9,} / "
              f"使える {100*good.mean():.0f}%", flush=True)

    pl.DataFrame(per_day).write_parquet(
        DATA / f"xvl_ccfday_{a.sym}{sfx}.parquet")
    D = pl.DataFrame(daily, infer_schema_length=None)
    D.write_csv(DATA / f"xvl_daily_{a.sym}{sfx}.csv")
    V = pl.DataFrame(devrows, infer_schema_length=None)
    V.write_csv(DATA / f"xvl_dev_{a.sym}{sfx}.csv")

    ok = D.filter(pl.col("ok"))
    print(f"\n有効 {ok.height} 日 / 使えるセル {ok['usable_pct'].mean():.1f}% "
          f"(Lighter 鮮度 {ok['lgt_fresh_pct'].mean():.1f}% / "
          f"Binance {ok['bnc_fresh_pct'].mean():.1f}%)")
    print(f"セル内で動く割合: Lighter {ok['lgt_move_pct'].mean():.1f}% / "
          f"Binance {ok['bnc_move_pct'].mean():.1f}%")
    print(f"打刻→手元の実測(中央): Lighter {ok['lgt_lat_ms_p50'].median():+.0f}ms"
          f" / Binance {ok['bnc_lat_ms_p50'].median():+.0f}ms"
          f"  ※取引所の時計のずれを含む生の差")
    print(f"gz の壊れたメンバー 合計 {int(ok['gz_bad'].sum())}")
    hl = V["half_life_ms"].drop_nulls().drop_nans()
    print(f"\n乖離 |d| 中央 {V['dev_abs_p50'].median():.2f}bp / "
          f"90% 点 {V['dev_abs_p90'].median():.2f}bp / "
          f"基差 {V['basis_p50'].median():+.2f}bp")
    print(f"半減期 中央 {hl.median():.0f}ms "
          f"(四分位 {hl.quantile(.25):.0f}〜{hl.quantile(.75):.0f}ms)")
    lags = np.arange(-nlag, nlag + 1) * a.cell
    names = ["全体"] + [s[0] for s in SESS]
    print(f"\n{'ラグ(ms)':>9s}  " + "".join(f"{n:>12s}" for n in names))
    print("  ★正のラグ = Binance が先、Lighter が後")
    for j, lg in enumerate(lags):
        if abs(lg) > 1000 and lg % 500:
            continue
        s = f"{int(lg):>9d}  "
        for i in range(len(names)):
            s += (f"{ccf_s[i, j]/ccf_n[i, j]:>12.4f}" if ccf_n[i, j]
                  else f"{'—':>12s}")
        print(s)


if __name__ == "__main__":
    main()
