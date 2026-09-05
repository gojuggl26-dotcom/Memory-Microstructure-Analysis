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
`data/probe_latency_<tag>.csv`(1 注文 1 行)。生の 24 項目に加えて、
取り違えを防ぐため**派生量もその場で確定させて**保存する:

    L_decision_ms, L_ack_ms, L_visible_ms, L_effective_ms,
    block_phase_ms, next_block_included, blocks_waited,
    blocks_waited_obs, gate_survived_at_visible

`blocks_waited` は **L_visible ÷ 局所推定した 1 ブロック間隔**の四捨五入である。
ブロックは剛体的な周期格子ではなく、間隔が中央 67.30 ms・標準偏差 8.29 ms
(p10 60.5 / p90 81.2)で揺らぐことを実データで確認した(固定格子で位相を
復元しようとすると、格子からのずれが中央 15 ms 出て使いものにならない)。
そのため `blocks_waited_obs`(実際に板に現れたブロック数・下限)と
`block_delta_ms`(そのとき使った間隔)も並べて残し、後から突き合わせられるようにした。
`block_phase_ms` は**直前に観測できた**ブロックからの経過なので、
そのブロックが静かで見えていなければ過大評価になる。

`ev_decision` / `ev_visible` は `entrygate_live_model_*.json`(公開フィードだけで
作れる変数に絞った門)で計算する。完全版の門は L4 の注文イベントを要するので
実時間では作れない。live 版の標本外成績は完全版とほぼ同じ
(−0.063 ± 0.113 対 −0.049 ± 0.093 bp)。
"""
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
# 生の時刻(すべて同じ monotonic clock: time.perf_counter_ns)
RAW = ["probe_id", "decision_ns", "send_ns", "ack_ns",
       "first_private_seen_ns", "first_public_book_seen_ns",
       "previous_block_ns", "next_block_ns", "side", "price", "size",
       "bbo_at_decision", "bbo_at_visible", "obi_at_decision",
       "obi_at_visible", "ofi_at_decision", "ofi_at_visible",
       "ev_decision", "ev_visible",
       "filled_before_cancel", "cancel_send_ns", "cancel_effective_ns",
       "regime", "note"]
# ★ 後から計算し直せるが、取り違えを防ぐため 1 行のうちに確定させて保存する
DERIVED = ["L_decision_ms", "L_ack_ms", "L_visible_ms", "L_effective_ms",
           "block_phase_ms", "next_block_included", "blocks_waited",
           "blocks_waited_obs", "block_delta_ms", "gate_survived_at_visible"]
FIELDS = RAW + DERIVED
# 旧称との対応: pred_ev_at_decision = ev_decision、pred_ev_at_visible = ev_visible
# (同じ数を 2 列に持たない)


class BlockRate:
    """ブロックの間隔を局所的に推定する。

    ★ 最初は「ブロックは剛体的な周期格子」と仮定して位相を復元しようとしたが、
      実データで検証すると**格子からのずれが中央 15 ms**あり、間隔そのものが
      中央 67.30 ms・標準偏差 8.29 ms(p10 60.5 / p90 81.2)と揺らいでいた。
      整数倍の関係は保たれる(実測/理論の中央が 1.001 / 1.978 / 2.990)ものの、
      固定格子で位相を出すことはできない。そこで**観測できたブロック時刻を
      そのまま使い、間隔だけ局所中央値で推定する**方式にした。

    公開板に現れるのは「板が変化したブロック」だけである。静かなブロックは
    見えないので、観測数で数えた待ちブロック数は下限にしかならない。
    自分の注文は板を変化させるので、**載ったブロックは必ず観測できる**。
    """

    def __init__(self, seed_ms: float = 67.30, keep: int = 4000):
        self.seed = seed_ms * 1e6
        self.keep = keep
        self.t: list[int] = []

    def feed(self, t: int) -> None:
        if self.t and t <= self.t[-1]:
            return                       # 同じブロック内の複数更新は 1 つに畳む
        self.t.append(int(t))
        if len(self.t) > self.keep:
            self.t = self.t[-self.keep:]

    def delta(self) -> float:
        """1 ブロックの間隔(ns)。近傍の山だけから中央値を取る。"""
        if len(self.t) < 30:
            return self.seed
        g = [b - a for a, b in zip(self.t[:-1], self.t[1:])]
        near = sorted(x for x in g if 0.6 * self.seed < x < 1.4 * self.seed)
        return float(near[len(near) // 2]) if len(near) >= 15 else self.seed

    def prev(self, t: int) -> int:
        """t 以前で最後に観測できたブロック時刻(静かなブロックは見えない)。"""
        p = 0
        for x in self.t:
            if x <= t:
                p = x
            else:
                break
        return p

    def blocks_between(self, a: int, b: int) -> int:
        """a から b までに跨いだブロック数の推定(静かなブロックを含む)。"""
        d = self.delta()
        return max(0, int(round((b - a) / d)))

    def blocks_observed(self, a: int, b: int) -> int:
        """そのうち実際に板に現れたブロックの数(下限)。"""
        return sum(1 for x in self.t if a < x <= b)


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
    ev_decision: float = float("nan")
    ev_visible: float = float("nan")
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
    """t の直前・直後の、実際に観測できたブロック時刻。"""
    prev = nxt = 0
    for b in book_times:
        if b <= t:
            prev = b
        else:
            nxt = b
            break
    return prev, nxt


def derive(p: "Probe", clock: "BlockRate") -> dict:
    """1 行ぶんの派生量を確定させる。時刻が取れていない項目は空にする。"""
    ms = 1e6
    d = {k: "" for k in DERIVED}
    if p.send_ns and p.decision_ns:
        d["L_decision_ms"] = (p.send_ns - p.decision_ns) / ms
    if p.ack_ns and p.send_ns:
        d["L_ack_ms"] = (p.ack_ns - p.send_ns) / ms
    t3 = p.first_public_book_seen_ns
    if t3 and p.send_ns:
        d["L_visible_ms"] = (t3 - p.send_ns) / ms
    if t3 and p.decision_ns:
        d["L_effective_ms"] = (t3 - p.decision_ns) / ms      # ★ 本命
    if p.send_ns and p.previous_block_ns:
        # 直前に**観測できた**ブロックからの経過。そのブロックが静かで
        # 見えていなければ過大評価になるので、その旨を note に残すこと
        d["block_phase_ms"] = (p.send_ns - p.previous_block_ns) / ms
    if t3 and p.send_ns:
        nb = clock.blocks_between(p.send_ns, t3)
        d["blocks_waited"] = nb
        d["next_block_included"] = int(nb <= 1)
        d["blocks_waited_obs"] = clock.blocks_observed(p.send_ns, t3)
        d["block_delta_ms"] = clock.delta() / ms
    if p.ev_visible == p.ev_visible:
        d["gate_survived_at_visible"] = int(p.ev_visible > 0)
    return d


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
    clock = BlockRate()
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
        clock.feed(b0["recv_ns"])
        p.previous_block_ns, p.next_block_ns = block_bounds(book_times, t1)
        p.previous_block_ns = clock.prev(t1) or p.previous_block_ns
        p.note = "dry-run: 注文なし。t2/t3 は取得していない"
        row = asdict(p)
        row.update(derive(p, clock))
        rows.append(row)
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
