r"""Derive オプション — ウォレット別の完全な損益台帳を組む。

=============================================================================
★実測で確定させた意味論(ここを外すと全部が嘘になる)
=============================================================================
オプションの建玉は **反対売買**か**満期決済**でしか終わらない。両方を足し、
まだ生きている銘柄の残存建玉を時価評価すれば漏れが無い。

  (a) 反対売買  `realized_pnl`(手数料差引後) / `realized_pnl_excl_fees`(手数料前)
      ★意味 = 平均取得単価に対する**約定ごとの実現損益**。
        ETH-20260911-2600-C / subaccount 51031 の全 194 約定を独立に再生して
        **非ゼロ行 10/10 で完全一致**(合計差 −0.00)を確認済み。
        ★期間途中から再生すると合わない。必ず初回約定から追うこと。
  (b) 満期決済  `option_settlement_pnl`
      ★これを落とすと満期まで持った建玉の損益が丸ごと消える。
        `realized_pnl` だけだとメイカー −$6.1M / テイカー +$13.3M と
        符号が不自然に非対称になる(実測)。
  (c) 残存建玉  生きている銘柄の建玉を直近 mark で評価。
      ★**未実現なので主判定に混ぜない。**別欄で開示する。
  (d) perp     ★オプション MM はデルタを perp でヘッジする。
      perp の `realized_pnl` を入れないと採算を測り損ねる。
      実測: メイカー主体 41 ウォレットのうち **5 つが perp も取引**しており、
      その中に最大の赤字ウォレットと最大の黒字ウォレットの両方が含まれる。
      ★残り 36 は venue 外でヘッジしているか、建玉間で相殺しているか、
        ヘッジしていない。**venue 外のヘッジは原理的に観測できない**ので、
        「Derive 上で見える損益」であることを必ず明示する。

=============================================================================
★満期決済 API の穴と、その埋め方
=============================================================================
`get_option_settlement_history` は **古い満期ほど欠落する**(実測: 2024-06〜08 で
27%、2026-07 以降は 1% 台)。刻みを変えた 4 巡の走査で 253,291 件まで集めたが
それでも残る。そこで:

  - API に記録があればそれを使う(取引所の確定値)
  - 無ければ **自前で計算して埋める**(`filled` フラグを立てる)
      決済損益 = 建玉 × (本源的価値 − 平均取得単価)
      本源的価値 = max(0, S−K) [コール] / max(0, K−S) [プット]
    ★この式が API の値を **95.83% で再現する**ことを 203,223 件で確認済み。
    残る 4% は 2024-01(約定履歴の起点)より前に建てられた建玉で、
    平均取得単価が再生できないもの。**埋めずに落とす**(`unreliable`)。

★★`float("37_25")` は **3725.0** を返す(Python が下線を桁区切りと解釈する)。
  Derive の行使価格は小数点を `_` で書く(HYPE-20260314-**37_25**-P = 37.25)。
  ここを踏むと本源的価値が 100 倍になり、決済損益が 30 万倍ずれる。実際に踏んだ。

=============================================================================
★二重計上を避ける規則
=============================================================================
- trades_rest は **1 約定が maker 行と taker 行の 2 行**。損益は各行がその当事者の
  ものなので **両方使う**。ただし**数量・名目を足すときは片側だけ**にする
- 決済は (subaccount, 銘柄) で一意
- `expected_rebate` が `realized_pnl` に含まれるかは不明なので**別建て**で持つ

出力: E:/Memory-derive/ledger_*.parquet
"""
from __future__ import annotations

import argparse
import glob
import io
import json
from collections import defaultdict
from pathlib import Path

import polars as pl
import zstandard as zstd

SRC = Path("E:/Derive-options/data")
OUT = Path("E:/Memory-derive")


def read_zst(path):
    with open(path, "rb") as fh:
        r = zstd.ZstdDecompressor().stream_reader(fh)
        for line in io.TextIOWrapper(r, encoding="utf-8"):
            if line.strip():
                yield json.loads(line)


def strike_of(ins: str) -> float:
    """★`_` は小数点。float() に直接渡すと桁区切りとして食われる。"""
    return float(ins.split("-")[2].replace("_", "."))


def f(x, k, d=0.0):
    try:
        return float(x[k])
    except (TypeError, ValueError, KeyError):
        return d


def main() -> int:
    ap = argparse.ArgumentParser()
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------------- 約定を読む(オプションのみ)----------------
    rows = []
    for p in sorted(glob.glob(str(SRC / "trades_rest" / "dt=*" / "*.jsonl.zst"))):
        cur = Path(p).name.split(".")[0]
        if cur.endswith("_perp"):
            continue                       # perp は別立てで読む
        day = p.replace("\\", "/").split("dt=")[1][:10]
        for x in read_zst(p):
            rows.append({
                "day": day, "cur": cur, "ts": int(x["timestamp"]),
                "ins": x["instrument_name"], "sub": int(x["subaccount_id"]),
                "wallet": x["wallet"], "role": x["liquidity_role"],
                "dir": x["direction"], "amt": f(x, "trade_amount"),
                "px": f(x, "trade_price"), "mark": f(x, "mark_price"),
                "index": f(x, "index_price"),
                "realized": f(x, "realized_pnl"),
                "realized_gross": f(x, "realized_pnl_excl_fees"),
                "fee": f(x, "trade_fee"), "rebate": f(x, "expected_rebate"),
                "rfq": x.get("rfq_id") is not None,
            })
    T = pl.DataFrame(rows)
    T.write_parquet(OUT / "ledger_trades.parquet")
    print(f"約定 {T.height:,} 行 / ウォレット {T['wallet'].n_unique():,} / "
          f"subaccount {T['sub'].n_unique():,} / {T['day'].min()}〜{T['day'].max()}")

    # ---------------- 建玉と平均取得単価を再生 ----------------
    # ★時刻順に舐める。部分決済は平均法。初回約定から追わないと合わない
    state = defaultdict(lambda: [0.0, 0.0])          # qty, cost
    first_day = {}
    for r in sorted(rows, key=lambda z: z["ts"]):
        k = (r["sub"], r["ins"])
        first_day.setdefault(r["ins"], r["day"])
        q = r["amt"] if r["dir"] == "buy" else -r["amt"]
        pos, cost = state[k]
        if pos != 0 and (pos > 0) != (q > 0):
            close = min(abs(q), abs(pos))
            avg = cost / abs(pos)
            cost -= close * avg
            pos -= close * (1 if pos > 0 else -1)
            rem = abs(q) - close
            if rem > 1e-12:
                pos, cost = rem * (1 if q > 0 else -1), rem * r["px"]
        else:
            pos += q
            cost += abs(q) * r["px"]
        state[k] = [pos, cost]
    print(f"建玉再生 {len(state):,} 組")

    # ---------------- 満期決済 ----------------
    S = {}
    settle_px = {}
    for x in read_zst(SRC / "settlements" / "settlements.jsonl.zst"):
        k = (int(x["subaccount_id"]), x["instrument_name"])
        S[k] = x
        settle_px[x["instrument_name"]] = f(x, "settlement_price")
    live = set()
    lp = SRC / "instruments" / "latest.json"
    if lp.exists():
        live = {y["name"] for y in json.loads(lp.read_text(encoding="utf-8"))}

    # ★約定履歴の起点。ここで既に取引が始まっていた銘柄は平均単価が信用できない
    D0 = T["day"].min()
    srows, filled, unreliable = [], 0, 0
    for (sub, ins), (q, cost) in state.items():
        rec = S.get((sub, ins))
        if rec is not None:
            srows.append({"sub": sub, "ins": ins, "cur": ins.split("-")[0],
                          "expiry": int(rec["expiry"]),
                          "pnl": f(rec, "option_settlement_pnl"),
                          "pnl_gross": f(rec, "option_settlement_pnl_excl_fees"),
                          "qty": f(rec, "amount"), "src": "api"})
            continue
        if abs(q) < 1e-9 or ins in live:
            continue                                  # 反対売買で閉じた/まだ生きている
        if first_day.get(ins, "9999") <= D0:
            unreliable += 1                           # 起点より前から動いていた銘柄
            continue
        sp = settle_px.get(ins)
        if sp is None:
            unreliable += 1
            continue
        K = strike_of(ins)
        v = max(0.0, sp - K) if ins.split("-")[3] == "C" else max(0.0, K - sp)
        pnl = q * (v - cost / abs(q))
        srows.append({"sub": sub, "ins": ins, "cur": ins.split("-")[0],
                      "expiry": 0, "pnl": pnl, "pnl_gross": pnl,
                      "qty": q, "src": "filled"})
        filled += 1
    SE = pl.DataFrame(srows)
    SE.write_parquet(OUT / "ledger_settle.parquet")
    n_api = int((SE["src"] == "api").sum())
    print(f"満期決済 {SE.height:,} 件 = API {n_api:,} + 自前補完 {filled:,} "
          f"/ 平均単価が再生できず落とした {unreliable:,}")

    # ---------------- 残存建玉(未実現・別建て)----------------
    last_mark = {}
    for r in sorted(rows, key=lambda z: z["ts"]):
        last_mark[r["ins"]] = r["mark"]
    orows = []
    for (sub, ins), (q, cost) in state.items():
        if abs(q) < 1e-9 or ins not in live:
            continue
        m = last_mark.get(ins, 0.0)
        orows.append({"sub": sub, "ins": ins, "cur": ins.split("-")[0],
                      "qty": q, "mark": m, "avg": cost / abs(q),
                      "mtm": q * (m - cost / abs(q))})
    O = pl.DataFrame(orows, schema={"sub": pl.Int64, "ins": pl.String,
                                    "cur": pl.String, "qty": pl.Float64,
                                    "mark": pl.Float64, "avg": pl.Float64,
                                    "mtm": pl.Float64})
    O.write_parquet(OUT / "ledger_open.parquet")
    print(f"残存建玉 {O.height:,} 組 / 含み損益 ${O['mtm'].sum():,.0f}(未実現)")

    # ---------------- perp(ヘッジ)----------------
    # ★perp は 290 万行ある。辞書のリストで全部持つと 8GB を超えて OOM する
    #   (RAM 15.3GB のこの機で実際に空き 0.2GB まで落ちた)。
    #   ファイルごとに polars へ渡し、**(wallet, day) の集計だけ**を貯める。
    #   生の perp 行は data/trades_rest/*_perp に残っているので必要なら後で読む。
    pagg = []
    nperp = 0
    for p in sorted(glob.glob(str(SRC / "trades_rest" / "dt=*" / "*_perp.jsonl.zst"))):
        cur = Path(p).name.split("_perp")[0]
        day = p.replace("\\", "/").split("dt=")[1][:10]
        buf = []
        for x in read_zst(p):
            buf.append((x["wallet"], x["liquidity_role"],
                        f(x, "realized_pnl"), f(x, "realized_pnl_excl_fees"),
                        f(x, "trade_fee"), f(x, "expected_rebate"),
                        f(x, "trade_amount") * f(x, "index_price")))
        if not buf:
            continue
        nperp += len(buf)
        d = pl.DataFrame(buf, schema=["wallet", "role", "realized",
                                      "realized_gross", "fee", "rebate",
                                      "notional"], orient="row")
        pagg.append(
            d.group_by(["wallet", "role"]).agg([
                pl.col("realized").sum(), pl.col("realized_gross").sum(),
                pl.col("fee").sum(), pl.col("rebate").sum(),
                pl.col("notional").sum(), pl.len().alias("n"),
            ]).with_columns([pl.lit(day).alias("day"), pl.lit(cur).alias("cur")]))
        del buf, d
    P = (pl.concat(pagg) if pagg else
         pl.DataFrame(schema={"wallet": pl.String, "role": pl.String,
                              "realized": pl.Float64,
                              "realized_gross": pl.Float64, "fee": pl.Float64,
                              "rebate": pl.Float64, "notional": pl.Float64,
                              "n": pl.UInt32, "day": pl.String,
                              "cur": pl.String}))
    P.write_parquet(OUT / "ledger_perp.parquet")
    print(f"perp 約定 {nperp:,} 行 -> (wallet, day, cur, role) 集計 {P.height:,} 行"
          + (f" / ウォレット {P['wallet'].n_unique():,}" if P.height else ""))
    del pagg

    # ---------------- subaccount -> wallet ----------------
    W = (T.group_by(["sub", "wallet"]).agg(pl.len().alias("n"))
         .sort("n", descending=True).unique(subset=["sub"], keep="first")
         .select(["sub", "wallet"]))
    W.write_parquet(OUT / "wallet_map.parquet")
    miss = SE.join(W, on="sub", how="left")["wallet"].null_count()
    print(f"wallet 対応 {W.height:,} / 決済のうち引けない {miss:,} "
          f"({miss/max(SE.height,1)*100:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
