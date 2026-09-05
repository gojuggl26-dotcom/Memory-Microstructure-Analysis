"""新規発注の実効遅延を測る probe(★ 既定では 1 件も注文を出さない)。

    # 何も送らずに配線と時刻計測だけ確かめる
    uv run python scripts/probe_latency.py --coin xyz:MU --n 20

    # 実際に注文を出す(要: 資金のある口座・取引権限のある API wallet・明示の同意)
    uv run python scripts/probe_latency.py --coin xyz:MU --n 2000 \
        --live --i-understand-this-sends-real-orders

測るもの
--------
    L_decision = t1 − t0   シグナル評価の完了から送信開始まで(自分側)
    L_ack      = t2 − t1   送信から取引所の受付確認まで(private stream)
    L_visible  = t3 − t1   送信から公開板に自分の注文が現れるまで(public stream)
    L_effective ≈ t3 − t0  ★ 戦略上これが本命

t3 は「板に入っていることを自分が確認できた時刻」なので、matching engine 上で
約定可能になった時刻の**保守的な上限**として扱う。t2 と t3 は独立した 2 経路
(private の注文状態と public の板)で測る。

block phase を必ず一緒に記録する
--------------------------------
この市場はブロック周期(約 65 ms)の構造が強い。平均値だけでは
「ネットワークが遅いのか、ブロック待ちなのか」を分離できない。そこで

    phi = t_send − (直前のブロック時刻)

を記録し、L_visible を phi に対して見る。1 ブロックで入るなら
L ≈ 65 − phi + ε、2 ブロック必要なら L ≈ 130 − phi + ε という鋸歯が出る。
併せて跨いだブロック数 N_block も数える。

安全装置
--------
- **既定は dry-run。`--live` と `--i-understand-this-sends-real-orders` の
  両方が無ければ 1 件も送らない**
- post-only(ALO)のみ。テイクは絶対にしない
- 数量は `--size`(既定は最小ロット)。`--max-notional-usd` を超える設定は拒否
- 1 件ごとに `--gap` 秒あける(既定 2 秒)。連打しない
- 送信後 `--ttl` 秒(既定 1 秒)で必ず取り消す
- `--n` の上限と、累計で `--max-orders` を超えたら停止
- 板の最良から `--depth` ティック内側には置かない(既定 2)。
  BBO での測定は `--depth 0` を明示したときだけ

依存(未導入)
--------------
    uv add hyperliquid-python-sdk websockets eth-account

取引所へ触る部分は `Exchange` として切り出してある。公式 SDK が署名を扱うので、
`LiveExchange` の 3 メソッドを SDK 呼び出しで埋めれば動く。
**この器は注文を送る経路を持たない状態で配布している。**

出力
----
`data/probe_latency_<tag>.csv`(1 注文 1 行)。列は指示された 24 項目。
"""
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIELDS = ["probe_id", "decision_ns", "send_ns", "ack_ns",
          "first_private_seen_ns", "first_public_book_seen_ns",
          "previous_block_ns", "next_block_ns", "side", "price", "size",
          "bbo_at_decision", "bbo_at_visible", "obi_at_decision",
          "obi_at_visible", "ofi_at_decision", "ofi_at_visible",
          "pred_ev_at_decision", "pred_ev_at_visible",
          "filled_before_cancel", "cancel_send_ns", "cancel_effective_ns",
          "regime", "note"]


@dataclass
class Probe:
    probe_id: int
    decision_ns: int = 0
    send_ns: int = 0
    ack_ns: int = 0
    first_private_seen_ns: int = 0
    first_public_book_seen_ns: int = 0
    previous_block_ns: int = 0
    next_block_ns: int = 0
    side: int = 1
    price: float = 0.0
    size: float = 0.0
    bbo_at_decision: str = ""
    bbo_at_visible: str = ""
    obi_at_decision: float = 0.0
    obi_at_visible: float = 0.0
    ofi_at_decision: float = 0.0
    ofi_at_visible: float = 0.0
    pred_ev_at_decision: float = 0.0
    pred_ev_at_visible: float = 0.0
    filled_before_cancel: int = 0
    cancel_send_ns: int = 0
    cancel_effective_ns: int = 0
    regime: str = ""
    note: str = ""


class Exchange:
    """取引所への口。dry-run は何も送らない。"""

    def __init__(self, live: bool):
        self.live = live

    def book(self) -> dict:
        """最新の板(最良気配と数量)。public stream から。"""
        raise NotImplementedError(
            "公開板の購読を実装すること。websockets で l2Book を購読し、"
            "受信時刻(perf_counter_ns)と内容を保持する。")

    def place(self, side: int, price: float, size: float, cloid: str) -> int:
        """post-only の指値を出し、受付確認の時刻を返す。"""
        raise NotImplementedError(
            "公式 SDK の exchange.order(..., order_type={'limit':{'tif':'Alo'}}) "
            "を呼ぶこと。cloid を必ず付けて一意に特定できるようにする。")

    def cancel(self, cloid: str) -> int:
        raise NotImplementedError("公式 SDK の exchange.cancel_by_cloid を呼ぶこと。")


class DryRun(Exchange):
    """注文を出さずに、時刻計測と block phase の記録だけ動かす。"""

    def __init__(self):
        super().__init__(False)
        self._t0 = time.perf_counter_ns()

    def book(self) -> dict:
        # 実際には public stream の最新スナップショットを返す。
        # ここでは配線確認のため、時刻だけ本物を使う。
        now = time.perf_counter_ns()
        return {"recv_ns": now, "bid": float("nan"), "ask": float("nan"),
                "bid_sz": float("nan"), "ask_sz": float("nan")}

    def place(self, side, price, size, cloid):
        return 0

    def cancel(self, cloid):
        return 0


def block_bounds(book_times, t):
    """t の直前・直後のブロック時刻。公開板の受信時刻の列から取る。"""
    prev = nxt = 0
    for b in book_times:
        if b <= t:
            prev = b
        else:
            nxt = b
            break
    return prev, nxt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    ap.add_argument("--n", type=int, default=20, help="probe の本数")
    ap.add_argument("--gap", type=float, default=2.0, help="1 件ごとの間隔(秒)")
    ap.add_argument("--ttl", type=float, default=1.0, help="出してから取り消すまで")
    ap.add_argument("--depth", type=int, default=2,
                    help="最良から何ティック外に置くか。0 は BBO(要注意)")
    ap.add_argument("--size", type=float, default=0.001)
    ap.add_argument("--max-notional-usd", type=float, default=50.0)
    ap.add_argument("--max-orders", type=int, default=10000)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--i-understand-this-sends-real-orders", action="store_true",
                    dest="confirm")
    a = ap.parse_args()

    if a.live and not a.confirm:
        raise SystemExit("--live には --i-understand-this-sends-real-orders が要る")
    if a.n > a.max_orders:
        raise SystemExit("--n が --max-orders を超えている")
    ex: Exchange = DryRun() if not a.live else Exchange(True)
    if a.live:
        raise SystemExit(
            "この器は取引所への送信経路を持たない状態で配布している。\n"
            "  1. uv add hyperliquid-python-sdk websockets eth-account\n"
            "  2. Exchange.book / place / cancel を SDK で実装する\n"
            "  3. 取引権限のある API wallet を用意する\n"
            "  4. まず --depth 2 の小口で数十件、次に --depth 0 へ\n"
            "実行の可否は人間が判断すること。")

    tag = a.coin.replace(":", "_")
    out = DATA / f"probe_latency_{tag}.csv"
    book_times: list[int] = []
    rows = []
    print(f"dry-run: {a.n} 件ぶんの時刻計測だけ回す(注文は 1 件も出さない)")
    for i in range(a.n):
        b0 = ex.book()
        t0 = time.perf_counter_ns()                       # シグナル評価完了
        p = Probe(probe_id=i, decision_ns=t0, side=1,
                  size=a.size, regime="dry")
        t1 = time.perf_counter_ns()                       # 送信開始
        p.send_ns = t1
        p.ack_ns = ex.place(1, 0.0, a.size, f"probe-{i}")
        book_times.append(b0["recv_ns"])
        p.previous_block_ns, p.next_block_ns = block_bounds(book_times, t1)
        p.note = "dry-run: 注文なし。t2/t3 は取得していない"
        rows.append(asdict(p))
        time.sleep(min(a.gap, 0.01))                      # dry では詰めて回す
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"書き出し {out}({len(rows)} 行・すべて dry-run)")
    print("L_decision の実測(自分側の処理時間):",
          f"中央 {sorted(r['send_ns'] - r['decision_ns'] for r in rows)[len(rows)//2]/1e3:.1f} us")


if __name__ == "__main__":
    main()
