"""板と注文イベントの「散らばり具合」をエントロピー 13 種で測る。

【何を測るか】
エントロピー H = −Σ p log₂ p は、確率分布がどれだけ「広がっているか」を測る。
1 点に集中していれば 0、k 個に均等なら log₂k で最大になる。単位はビット。
ここでは板の状態と注文イベントの流れについて 13 種類を 1 秒格子で出す。

  板のエントロピー(その瞬間に板にある指値注文が母集団)
    h_depth    価格水準ごとの**数量**の散らばり     p_i = vol_i / Σvol
    h_size     注文数量の**大きさ階級**の散らばり   p_b = n_b / N(log₂ 階級)
    h_plevel   価格水準ごとの**本数**の散らばり     p_i = n_i / N
    h_wallet   口座ごとの**数量**の散らばり         p_w = vol_w / Σvol
    h_age      注文の**滞留時間階級**の散らばり     p_b = n_b / N(対数階級)
    h_queue    最良気配の**待ち行列内**の散らばり   p_k = sz_k / Σsz(両側の数量加重平均)

  イベントのエントロピー(直前 1 秒に起きたイベントが母集団)
    h_action   NEW / REMOVE / UPDATE の 3 値
    h_side     bid / ask の 2 値(最大 1 ビット)
    h_dir      板を押し上げる向き / 押し下げる向きの 2 値
               上 = bid に置く or ask を消す、下 = ask に置く or bid を消す

  動的なエントロピー
    h_chg      h_depth(T) − h_depth(T−1s)。板の散らばりの増減
    h_rate     エントロピー率 H₂/2(直前 10 秒、記号 = 動作 × 側の 6 値)
    h_cond     条件付きエントロピー H(Xₜ|Xₜ₋₁) = H₂ − H₁(同上)
    h_trans    遷移エントロピー。観測された行 i について H(次|現在=i) を
               **出現頻度で重みづけせず**平均したもの

★ h_rate / h_cond / h_trans は互いに強く関係する量である(いずれも同じ
  2 記号ブロック分布から作る)。別物として扱わず、レポートで相関を実測して示す。
  h_trans だけは行を占有確率で重みづけしないので h_cond とは一致しない
  (「よく出る状態」ではなく「典型的な状態」の出口の読みにくさを測る)。

【x が確定する時刻 / y の期間】★先読みなし
    x = 上の 13 種。格子時刻 T について
        板の 6 種  … ts <  T のイベントだけを適用した板の状態(同一 ns の
                     自分自身を含めないため厳密に <)
        イベント 3 種 … [T−1s, T) に起きたイベント
        動的 4 種  … h_chg は T と T−1s の板、残りは [T−10s, T) のイベント
    y = [T, T+h) の mid の対数リターン(h = 1, 2, 5, 10, 30, 60 秒)。
        向き r と大きさ |r| の両方を目的変数にする。
先読みは無い。asof は backward のみ(bbo は「厳密に T 未満の最後の行」)。

【★踏んだ落とし穴】
- **遠方に駐機している注文が板の統計を支配する。** 発注の距離分布は mid から
  p90 が 64.5bp なのに p95 で 999bp へ跳ぶ。1000bp 付近に別の群がある。
  これを入れると h_wallet / h_plevel がその群だけで決まるので、板の 6 種は
  **mid ±100bp の帯**に入る注文だけを母集団にする(発注の 90.8% を覆う)。
- **ティック幅は 0.01 固定ではない**(1000 以上は 0.1)。ただし本script は
  価格を刻み幅で割らず「占有された水準」を数えるだけなので影響を受けない。
- **クロスした板**(best_ask ≤ best_bid)と bbo の異常行は除外する。
  少数の異常行が係数を大きく動かした前例がある(clean_bbo を通す)。
- **欠測時間帯**は格子ごと null にする。欠測中は板が凍って見えるので、
  エントロピーが「一定」に化ける。欠測の所在はサイドカー
  (`l1_meta_<coin>.csv` の missing_hours)を正とする。
- **トリガー注文**は発火まで板に載らないので母集団から外す。
  **テイカー**(Ioc など)は板に留まらないので Alo / Gtc だけを見る。
- 板が空/1 本だけの瞬間はエントロピーが定義できない(または常に 0)。
  本数がしきい値未満の格子は null にする。

【★活動量の統制】
エントロピーは母集団が大きいほど機械的に上がる(k 個に均等なら log₂k)。
「エントロピーが効く」が単に「注文が多い日は荒れる」の言い換えでないことを
確かめるため、母集団の大きさそのものも出力する。

    n_ord   格子時刻 T に帯の中で板にある注文の本数
    n_ev    [T−1s, T) に起きたイベント数

【分割実行と再開】
1 日あたり約 100 秒かかるので、日ごとに書き出して途中から再開できるようにする。
`--start` / `--n` で日を区切れば並列に走らせられる。分割すると板の持ち越しが
途切れるため、`--warmup` 日ぶんを**書き出さずに処理して板を温めてから**始める。
温めが足りるかは実測で確かめること(`--warmup 0` と比べて差が出ないこと)。

    # 4 並列の例
    for s in 0 25 50 75; do
      python scripts/build_entropy.py --coin xyz:MU --start $s --n 25 &
    done; wait
    python scripts/build_entropy.py --coin xyz:MU --merge

【出力】
    data/entropy_days_<coin>/dt=*.parquet  日ごとの中間ファイル(再開の単位)
    data/entropy_<coin>.parquet    1 秒格子 × 13 特徴量 + 統制 2 + mid + 将来リターン
    data/entropy_meta_<coin>.csv   日ごとの検算(格子数・欠測・板が薄い格子)

実行例:
    uv run python scripts/build_entropy.py --coin xyz:MU
    uv run python scripts/build_entropy.py --selftest    # 計算の自己検証だけ実行
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import (CANCELS, DAY_NS, PX_UNIT, RESTING,  # noqa: E402
                              TERMINAL, clean_bbo)

ROOT = Path(__file__).resolve().parents[1]

GRID_NS = 1_000_000_000                  # 1 秒格子
BAND_BP = 100.0                          # 板の母集団(mid からの片側 bp)
EV_WIN_NS = 1_000_000_000                # イベント 3 種の窓(1 秒)
SEQ_WIN_NS = 10_000_000_000              # 動的 3 種の窓(10 秒)
MIN_ORDERS = 3                           # これ未満の板はエントロピーを出さない
MIN_EVENTS = 5                           # これ未満のイベント窓も出さない
HORIZONS = (1, 2, 5, 10, 30, 60)         # 将来リターンの地平(秒)
WARMUP_DAYS = 5                          # 分割実行で板を温める日数(§下記で実測)

# 注文数量の階級(log₂)。実測 p50=4 / p99=98 契約を覆う
SIZE_EDGES = np.array([2.0 ** k for k in range(-3, 13)], dtype=np.float64)
# 滞留時間の階級(ns)。1ms 未満から 1 日超まで
AGE_EDGES = np.array([1e6, 1e7, 1e8, 1e9, 1e10, 6e10, 6e11,
                      3.6e12, 2.16e13, 8.64e13], dtype=np.float64)

FEATURES = ("h_depth", "h_size", "h_plevel", "h_wallet", "h_age", "h_queue",
            "h_action", "h_side", "h_dir",
            "h_chg", "h_rate", "h_cond", "h_trans")


def entropy(counts: np.ndarray) -> float:
    """Shannon エントロピー(ビット)。counts は非負の重み。"""
    tot = counts.sum()
    if tot <= 0:
        return np.nan
    p = counts[counts > 0] / tot
    return float(-(p * np.log2(p)).sum())


def entropy_from_labels(lab: np.ndarray, w: np.ndarray | None, k: int) -> float:
    """ラベル配列(0..k-1)から重みつきエントロピーを出す。"""
    c = np.bincount(lab, weights=w, minlength=k)
    return entropy(c)


def grid_mid(bb: pl.DataFrame, t0: int, ng: int):
    """bbo を格子へ後ろ向きに載せる。厳密に T 未満の最後の行を使う。"""
    ts = bb["ts"].cast(pl.Int64).to_numpy()
    tg = t0 + np.arange(ng, dtype=np.int64) * GRID_NS
    j = np.searchsorted(ts, tg, side="left") - 1        # ★ T 未満(先読みしない)
    ok = j >= 0
    j = np.where(ok, j, 0)
    pb = bb["best_bid"].to_numpy()[j]
    pa = bb["best_ask"].to_numpy()[j]
    good = ok & np.isfinite(pb) & np.isfinite(pa) & (pa > pb)   # クロス除外
    mid = np.where(good, (pb + pa) / 2.0, np.nan)
    bi = np.where(good, np.rint(pb / PX_UNIT), 0).astype(np.int64)
    ai = np.where(good, np.rint(pa / PX_UNIT), 0).astype(np.int64)
    return mid, bi, ai, good


def load_day(fp: Path, userdir: Path):
    """1 日分の L1 を、板の再生に必要な形へ整える。

    口座は `l1user_<coin>/dt=*.parquet`(oid → wid の対応表)を oid で結合する。
    行の並びに依存しないので、この経路が最も安全である。
    """
    lf = pl.scan_parquet(fp)
    n_raw = int(lf.select(pl.len()).collect().item())
    # ★必要な列だけを読む。1 日 3,900 万行の日があるので全列だと入らない
    keep = ((~pl.col("is_trigger")) & pl.col("tif").is_in(RESTING)
            & pl.col("status").is_in(["open"] + TERMINAL))
    d = lf.filter(keep).select(
        "oid", "ts", "side", "px", "status", "remaining_sz").collect()
    uf = userdir / fp.name if userdir is not None else None
    if uf is not None and uf.exists():
        d = d.join(pl.read_parquet(uf).unique(subset=["oid"]), on="oid", how="left")
    if "wid" not in d.columns:
        d = d.with_columns(pl.lit(-1, pl.Int32).alias("wid"))
    d = d.with_columns(pl.col("wid").fill_null(-1).cast(pl.Int32))
    ev = d.select(
        oid=pl.col("oid").cast(pl.Int64),
        ts=pl.col("ts").cast(pl.Int64),
        is_bid=(pl.col("side") == "B"),
        pidx=(pl.col("px") / PX_UNIT).round().cast(pl.Int64),
        px=pl.col("px"),
        # 取消なら 0、それ以外は残量。'filled' で残量>0 は部分約定(= UPDATE)
        size_after=pl.when(pl.col("status").is_in(CANCELS)).then(0.0)
                     .otherwise(pl.col("remaining_sz")),
        is_open=(pl.col("status") == "open"),
        is_cancel=pl.col("status").is_in(CANCELS),
        wid=pl.col("wid"),
    )
    # 同一 ns 内は open → 部分約定 → 終端 の論理順で並べる
    ev = ev.with_columns(
        rk=pl.when(pl.col("is_open")).then(0)
           .when(pl.col("size_after") > 0).then(1).otherwise(2).cast(pl.Int8)
    ).sort(["ts", "oid", "rk"])
    return ev, n_raw


def day_features(ev: pl.DataFrame, bb_day: pl.DataFrame, t0: int, ng: int,
                 carry: dict, miss_h: list[int]):
    """1 日分の 13 特徴量を 1 秒格子で返す。carry は前日から生きている注文。"""
    mid, bbi, bai, good = grid_mid(bb_day, t0, ng)

    # --- 注文ごとの静的属性を slot 配列に持つ(この日に触れた注文 + 繰越) ---
    oids = ev["oid"].unique(maintain_order=True)
    slot_of = {o: i for i, o in enumerate(oids.to_list())}
    n_new = len(slot_of)
    c_oids = [o for o in carry.get("oid", []) if o not in slot_of]
    for o in c_oids:
        slot_of[o] = len(slot_of)
    ns = len(slot_of)

    a_pidx = np.zeros(ns, np.int64)
    a_px = np.zeros(ns, np.float64)
    a_sz = np.zeros(ns, np.float64)
    a_bid = np.zeros(ns, np.bool_)
    a_open = np.full(ns, t0, np.int64)
    a_wid = np.full(ns, -1, np.int32)

    act: set[int] = set()
    if carry.get("oid"):
        idx = np.array([slot_of[o] for o in carry["oid"]], dtype=np.int64)
        a_pidx[idx] = carry["pidx"]
        a_px[idx] = carry["px"]
        a_sz[idx] = carry["sz"]
        a_bid[idx] = carry["is_bid"]
        a_open[idx] = carry["open_ts"]
        a_wid[idx] = carry["wid"]
        act.update(int(i) for i in idx)

    # --- イベントを格子ごとに切る ---
    e_ts = ev["ts"].to_numpy()
    e_slot = np.array([slot_of[o] for o in ev["oid"].to_list()], dtype=np.int64)
    e_gi = np.clip((e_ts - t0) // GRID_NS, 0, ng - 1).astype(np.int64)
    e_pidx = ev["pidx"].to_numpy()
    e_px = ev["px"].to_numpy()
    e_bid = ev["is_bid"].to_numpy()
    e_sa = ev["size_after"].to_numpy()
    e_op = ev["is_open"].to_numpy()
    e_cx = ev["is_cancel"].to_numpy()
    e_wid = ev["wid"].to_numpy()
    bnd = np.searchsorted(e_gi, np.arange(ng + 1, dtype=np.int64), side="left")

    out = {f: np.full(ng, np.nan) for f in FEATURES}
    n_ord = np.zeros(ng, np.int32)      # 帯の中の板の本数(統制変数)
    n_ev = np.zeros(ng, np.int32)       # 直前 1 秒のイベント数(統制変数)
    # 動作記号: 0=NEW 1=REMOVE 2=UPDATE、側と組み合わせて 6 値
    sym_buf: list[np.ndarray] = []      # 直近 SEQ_WIN 個の格子ぶんの記号列
    seq_len = int(SEQ_WIN_NS // GRID_NS)
    ev_buf: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    ev_len = int(EV_WIN_NS // GRID_NS)
    prev_hdepth = np.nan
    n_thin = 0

    miss = np.zeros(ng, np.bool_)
    for h in miss_h:
        miss[h * 3600:(h + 1) * 3600] = True

    for g in range(ng):
        s0, s1 = int(bnd[g]), int(bnd[g + 1])
        # ---- (A) この格子区間のイベントを板へ適用する ----
        if s1 > s0:
            sl = e_slot[s0:s1]
            op = e_op[s0:s1]
            if op.any():
                o = sl[op]
                a_pidx[o] = e_pidx[s0:s1][op]
                a_px[o] = e_px[s0:s1][op]
                a_bid[o] = e_bid[s0:s1][op]
                a_open[o] = e_ts[s0:s1][op]
                a_wid[o] = e_wid[s0:s1][op]
            a_sz[sl] = e_sa[s0:s1]
            for i, v in zip(sl.tolist(), e_sa[s0:s1].tolist()):
                if v > 0:
                    act.add(i)
                else:
                    act.discard(i)
            # 記号列(動作 × 側)
            act_lab = np.where(op, 0, np.where(e_sa[s0:s1] > 0, 2, 1))
            sym = (act_lab * 2 + e_bid[s0:s1].astype(np.int64)).astype(np.int64)
            ev_buf.append((act_lab, e_bid[s0:s1], e_sa[s0:s1] > 0))
            sym_buf.append(sym)
        else:
            ev_buf.append((np.empty(0, np.int64), np.empty(0, np.bool_),
                           np.empty(0, np.bool_)))
            sym_buf.append(np.empty(0, np.int64))
        if len(sym_buf) > seq_len:
            sym_buf.pop(0)
        if len(ev_buf) > ev_len:
            ev_buf.pop(0)

        # 格子 T の状態は「ts < T のイベントだけ」= ここまでの適用結果。
        # よって特徴量は次の格子 g+1 の時刻に対応する。ずれを避けるため
        # 出力は g+1 に書く(最後の格子ぶんは翌日へ繰り越す)。
        gw = g + 1
        if gw >= ng:
            continue
        if miss[gw] or not good[gw]:
            prev_hdepth = np.nan
            continue

        # ---- (B) 板のエントロピー(mid ±BAND_BP の帯) ----
        m = mid[gw]
        if act:
            ai = np.fromiter(act, np.int64, len(act))
            px = a_px[ai]
            inb = np.abs(px - m) / m * 1e4 <= BAND_BP
            ai = ai[inb]
        else:
            ai = np.empty(0, np.int64)
        n_ord[gw] = ai.size
        if ai.size >= MIN_ORDERS:
            sz = a_sz[ai]
            pidx = a_pidx[ai]
            # 価格水準を 0..k-1 に詰め直す
            _, lv = np.unique(pidx, return_inverse=True)
            k = lv.max() + 1
            out["h_depth"][gw] = entropy_from_labels(lv, sz, k)
            out["h_plevel"][gw] = entropy_from_labels(lv, None, k)
            sb = np.searchsorted(SIZE_EDGES, sz, side="right")
            out["h_size"][gw] = entropy(np.bincount(sb, minlength=SIZE_EDGES.size + 1))
            w = a_wid[ai]
            if (w >= 0).any():
                ww = w[w >= 0]
                _, wl = np.unique(ww, return_inverse=True)
                out["h_wallet"][gw] = entropy_from_labels(
                    wl, sz[w >= 0], wl.max() + 1)
            age = (t0 + gw * GRID_NS) - a_open[ai]
            ab = np.searchsorted(AGE_EDGES, age.astype(np.float64), side="right")
            out["h_age"][gw] = entropy(np.bincount(ab, minlength=AGE_EDGES.size + 1))
            # 待ち行列: 最良気配にいる注文の数量配分。両側を数量で重みづけ平均
            hq, wq = 0.0, 0.0
            for side_bid, bi in ((True, bbi[gw]), (False, bai[gw])):
                q = ai[(a_bid[ai] == side_bid) & (a_pidx[ai] == bi)]
                if q.size >= 2:
                    s = a_sz[q]
                    hq += entropy(s) * s.sum()
                    wq += s.sum()
            out["h_queue"][gw] = hq / wq if wq > 0 else 0.0
        else:
            n_thin += 1

        # ---- (C) イベントのエントロピー(直前 1 秒) ----
        al = np.concatenate([b[0] for b in ev_buf]) if ev_buf else np.empty(0, np.int64)
        n_ev[gw] = al.size
        if al.size >= MIN_EVENTS:
            bl = np.concatenate([b[1] for b in ev_buf]).astype(np.int64)
            alive = np.concatenate([b[2] for b in ev_buf])
            out["h_action"][gw] = entropy(np.bincount(al, minlength=3))
            out["h_side"][gw] = entropy(np.bincount(bl, minlength=2))
            # 上向き = bid に置く(NEW かつ bid) or ask を消す(REMOVE かつ ask)
            add = al == 0
            rem = al == 1
            up = (add & (bl == 1)) | (rem & (bl == 0))
            dn = (add & (bl == 0)) | (rem & (bl == 1))
            sel = up | dn
            if sel.sum() >= MIN_EVENTS:
                out["h_dir"][gw] = entropy(
                    np.array([up.sum(), dn.sum()], dtype=np.float64))
            del alive

        # ---- (D) 動的 ----
        hd = out["h_depth"][gw]
        out["h_chg"][gw] = hd - prev_hdepth
        prev_hdepth = hd

        sq = np.concatenate(sym_buf) if sym_buf else np.empty(0, np.int64)
        if sq.size >= MIN_EVENTS + 1:
            c1 = np.bincount(sq[:-1], minlength=6).astype(np.float64)
            pair = np.bincount(sq[:-1] * 6 + sq[1:], minlength=36).astype(np.float64)
            h1, h2 = entropy(c1), entropy(pair)
            out["h_rate"][gw] = h2 / 2.0
            out["h_cond"][gw] = h2 - h1
            M = pair.reshape(6, 6)
            rows = [entropy(M[i]) for i in range(6) if M[i].sum() > 0]
            out["h_trans"][gw] = float(np.mean(rows)) if rows else np.nan

    # --- 翌日へ渡す ---
    if act:
        ai = np.fromiter(act, np.int64, len(act))
        inv = {v: k for k, v in slot_of.items()}
        nxt = {"oid": [inv[int(i)] for i in ai], "pidx": a_pidx[ai],
               "px": a_px[ai], "sz": a_sz[ai], "is_bid": a_bid[ai],
               "open_ts": a_open[ai], "wid": a_wid[ai]}
    else:
        nxt = {}

    meta = {"n_grid": ng, "n_good": int(good.sum()), "n_miss": int(miss.sum()),
            "n_thin": n_thin, "n_new_oid": n_new, "n_carry_out": len(act),
            "med_orders": float(np.median(n_ord[good])) if good.any() else 0.0}
    return out, mid, good & ~miss, n_ord, n_ev, nxt, meta


def selftest() -> int:
    """エントロピーの計算を手計算できる既知の値と突き合わせる。

    ここが狂うと 13 種すべてが静かに狂う。数字を出す前に必ず通す。
    """
    cases = [
        ("1 点に集中 -> 0 bit", entropy(np.array([5.0])), 0.0),
        ("2 個均等 -> 1 bit", entropy(np.array([1.0, 1.0])), 1.0),
        ("4 個均等 -> 2 bit", entropy(np.array([1.0] * 4)), 2.0),
        ("8 個均等 -> 3 bit", entropy(np.array([1.0] * 8)), 3.0),
        ("偏り [3,1]", entropy(np.array([3.0, 1.0])), 0.8112781244591328),
        ("ゼロ要素は無視", entropy(np.array([1.0, 1.0, 0.0])), 1.0),
        ("尺度不変(×100)", entropy(np.array([300.0, 100.0])), 0.8112781244591328),
        ("重みなし 2 値均等",
         entropy_from_labels(np.array([0, 0, 1, 1]), np.ones(4), 2), 1.0),
        ("数量加重で [3,1]",
         entropy_from_labels(np.array([0, 0, 1, 1]),
                             np.array([2.0, 1.0, 0.5, 0.5]), 2),
         0.8112781244591328),
    ]
    bad = 0
    for name, got, want in cases:
        ok = abs(got - want) < 1e-9
        bad += not ok
        print(f"  {'OK ' if ok else 'NG '} {name}: {got!r}", file=sys.stderr)
    if not np.isnan(entropy(np.array([0.0, 0.0]))):
        print("  NG  総和 0 は nan であるべき", file=sys.stderr)
        bad += 1
    else:
        print("  OK  総和 0 -> nan", file=sys.stderr)
    print(f"自己検証: {'すべて OK' if not bad else f'★{bad} 件失敗'}",
          file=sys.stderr)
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    ap.add_argument("--days", type=int, default=0,
                    help="先頭から何日だけ処理するか(0 = 全部)")
    ap.add_argument("--start", type=int, default=0,
                    help="書き出しを始める日の番号(0 起点)")
    ap.add_argument("--n", type=int, default=0,
                    help="書き出す日数(0 = 最後まで)。--start と組で分割実行する")
    ap.add_argument("--warmup", type=int, default=WARMUP_DAYS,
                    help="--start の手前で板を温めるために処理するが書き出さない日数")
    ap.add_argument("--force", action="store_true",
                    help="既に書き出し済みの日も作り直す")
    ap.add_argument("--merge", action="store_true",
                    help="日ごとのファイルを 1 本の parquet にまとめて終了する")
    ap.add_argument("--selftest", action="store_true",
                    help="エントロピー計算の自己検証だけを行って終了する")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(1 if selftest() else 0)
    tag = a.coin.replace(":", "_")
    daydir = ROOT / "data" / f"entropy_days_{tag}"

    if a.merge:
        fs = sorted(daydir.glob("dt=*.parquet"))
        if not fs:
            sys.exit(f"{daydir} に日別ファイルがない")
        out = pl.concat([pl.read_parquet(f) for f in fs])
        out.write_parquet(ROOT / "data" / f"entropy_{tag}.parquet",
                          compression="zstd")
        ms = sorted(daydir.glob("meta=*.json"))
        pl.DataFrame([json.loads(m.read_text(encoding="utf-8")) for m in ms])           .write_csv(ROOT / "data" / f"entropy_meta_{tag}.csv")
        print(f"[merge] {len(fs)} 日 / {out.height:,} 行 "
              f"-> data/entropy_{tag}.parquet", file=sys.stderr)
        return

    files = sorted((ROOT / "data" / f"l1_{tag}").glob("dt=*.parquet"))
    if a.days:
        files = files[: a.days]
    # bbo にその日が無いと格子が作れないので、日の一覧だけ先に見る
    have = set(pl.scan_parquet(ROOT / "data" / f"bbo_{tag}.parquet")
               .select("dt").unique().collect()["dt"].to_list())
    files = [f for f in files if f.stem.split("=")[1] in have]

    # --- 分割実行。--start の手前 --warmup 日は板を温めるだけで書き出さない ---
    lo = max(0, a.start - a.warmup)
    hi = len(files) if not a.n else min(len(files), a.start + a.n)
    emit_from = a.start
    files = files[lo:hi]
    # ★担当する日の bbo だけを読む(全期間だと 3,500 万行で並列時に入らない)
    my_days = [f.stem.split("=")[1] for f in files]
    bb_all, n_drop = clean_bbo(
        pl.scan_parquet(ROOT / "data" / f"bbo_{tag}.parquet")
        .filter(pl.col("dt").is_in(my_days)).collect())

    userdir = ROOT / "data" / f"l1user_{tag}"
    n_wal = 0
    if (userdir / "_wallets.parquet").exists():
        n_wal = pl.read_parquet(userdir / "_wallets.parquet").height
    daydir.mkdir(parents=True, exist_ok=True)
    print(f"[load] 処理 {len(files)} 日(うち温め {a.start - lo} 日)/ "
          f"bbo 除外 {n_drop:,} 行 / 口座 {n_wal:,}", file=sys.stderr)

    miss_all: dict[str, list[int]] = {}
    metaf = ROOT / "data" / f"l1_meta_{tag}.csv"
    if metaf.exists():
        for r in pl.read_csv(metaf).iter_rows(named=True):
            h = json.loads(r["missing_hours"] or "null") or []
            miss_all[r["dt"]] = [int(x) for x in h if isinstance(x, int)]

    carry: dict = {}
    for k, fp in enumerate(files, start=lo):
        dt = fp.stem.split("=")[1]
        emit = k >= emit_from
        outf = daydir / f"dt={dt}.parquet"
        if emit and outf.exists() and not a.force:
            # 既にある日でも carry を繋ぐため計算自体は行う必要がある。
            # ただし温め区間を跨いで再開できるよう、書き出しだけ省く。
            print(f"  {dt} 済(スキップせず carry のみ更新)", file=sys.stderr)
        ev, n_raw = load_day(fp, userdir)
        if not ev.height:
            continue
        t0 = (int(ev["ts"].min()) // DAY_NS) * DAY_NS
        ng = DAY_NS // GRID_NS
        bb_day = bb_all.filter(pl.col("dt") == dt).sort("ts")
        cols, mid, ok, n_ord, n_ev, carry, meta = day_features(
            ev, bb_day, t0, ng, carry, miss_all.get(dt, []))
        if not emit:
            print(f"  {dt} 温め(書き出さない)", file=sys.stderr)
            continue
        meta["dt"] = dt
        meta["n_raw"] = n_raw
        meta["warmup_days"] = a.start - lo
        df = pl.DataFrame({"ts": t0 + np.arange(ng, dtype=np.int64) * GRID_NS,
                           "mid": mid, "ok": ok,
                           "n_ord": n_ord, "n_ev": n_ev,
                           **{k2: v for k2, v in cols.items()}})
        df = df.with_columns(pl.lit(dt).alias("dt"))
        # 将来リターン(y)。日をまたがせない
        df = df.with_columns([
            (np.log(pl.col("mid").shift(-h) / pl.col("mid"))).alias(f"r{h}")
            for h in HORIZONS])
        num = [c for c in df.columns
               if c not in ("ts", "dt", "ok", "n_ord", "n_ev")]
        df = df.with_columns([pl.col(c).cast(pl.Float32) for c in num])
        df.write_parquet(outf, compression="zstd")
        (daydir / f"meta={dt}.json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        print(f"  {dt} 良格子 {meta['n_good']:,} 欠測 {meta['n_miss']:,} "
              f"薄い {meta['n_thin']:,} 板の本数(中央) {meta['med_orders']:.0f} "
              f"繰越 {meta['n_carry_out']}", file=sys.stderr)

    print(f"[done] -> {daydir}", file=sys.stderr)


if __name__ == "__main__":
    main()
