"""発注を決めてから板に載るまでに、判断の根拠がどれだけ壊れるか(signal decay)。

    uv run python scripts/build_decay.py --coin xyz:MU

なぜ probe より先にこれをやるか
--------------------------------
本当に知りたいのはネットワークの ms ではなく、

    t0 で EV>0 と判断して発注し、t3 = t0 + L に板へ載ったとき、
    まだ EV>0 の状態か

である。これは**過去データだけで完全に測れる**。ここで L=50ms でも条件が
半分しか残らないなら、実際に probe を出して ms を測る前に方向が決まる。

各遅延 L について、門を通った発注時点 t0 に対し

    P(EV_{t0+L} > 0 | EV_{t0} > 0)         条件の生存率
    E[EV_{t0+L} − EV_{t0}]                  EV の減衰
    E[side·(mid_{t0+L} − mid_{t0})/mid·1e4] 自分に不利な向きの mid の動き
    E[OBI^{t0+L} − OBI^{t0}]、OFI も同様

を出す。t0+L の状態は **t0+L 以前の最後の板更新**(後ろ向き asof)で読む。
これは実際にその時刻に見える状態そのものである。

時間契約: t0 の判断は t0 までの情報のみ。t0+L の評価も t0+L までの情報のみ。
比較しているのは「同じ規則を 2 つの時刻で評価した結果」であって、未来は使わない。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_entrygate import COLS, ev_of, fit_models  # noqa: E402
from build_fillpnl import BULK, DATA, TRAIN_FRAC, design  # noqa: E402

LAGS = [0.010, 0.025, 0.050, 0.065, 0.100, 0.130, 0.200, 0.300, 0.500]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    ap.add_argument("--sfx", default="_q1")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    P = pl.read_parquet(DATA / f"inv_posts_{tag}{a.sfx}.parquet")
    files = sorted((BULK / tag).glob("dt=*.parquet"))
    ntr = int(round(len(files) * TRAIN_FRAC))
    mf = DATA / f"entrygate_model_{tag}{a.sfx}.json"
    if mf.exists():
        M = json.load(open(mf))
        mu = np.array(M["mu"]); sd_ = np.array(M["sd"])
        w_fill = np.array(M["w_fill"]); w_pnl = np.array(M["w_pnl"])
        print("学習済みの係数を読み込んだ", flush=True)
    else:
        mu, sd_, w_fill, w_pnl = fit_models(files, ntr, P, tag, a.sfx)

    acc = {L: {"n": 0, "surv": 0, "dev": 0.0, "dmid": 0.0, "dobi": 0.0,
               "dofi": 0.0, "ev0": 0.0, "ev1": 0.0} for L in LAGS}
    day_rows = []
    for k, f in enumerate(files[ntr:]):          # 評価期間だけ
        dt = f.stem.split("=")[1]
        # COLS には obi1 / ofi_0.1s が既に入っているので重複させない
        need = COLS + [c for c in ("mid", "obi1", "ofi_0.1s") if c not in COLS]
        T = pl.read_parquet(f, columns=need)
        X = design(T)
        ts = T["ts"].cast(pl.Int64).to_numpy()
        sdv = T["side"].to_numpy()
        mid = T["mid"].to_numpy()
        # side を掛けた mid の動き。正なら「決めてから載るまでに、
        # 買おう(売ろう)とした側へ価格が動いた」= 追いかける側になる
        ob = T["obi1"].to_numpy()
        of = T["ofi_0.1s"].to_numpy()
        for sg in (1, -1):
            m = np.flatnonzero(sdv == sg)
            tt = ts[m]
            ev0 = ev_of(X[m], mu, sd_, w_fill, w_pnl)
            g = ev0 > 0                          # 門を通る時点
            if not g.any():
                continue
            gi = np.flatnonzero(g)
            for L in LAGS:
                j = np.clip(np.searchsorted(tt, tt[gi] + int(L * 1e9),
                                            side="right") - 1, 0, tt.size - 1)
                e1 = ev_of(X[m][j], mu, sd_, w_fill, w_pnl)
                A = acc[L]
                A["n"] += gi.size
                A["surv"] += int((e1 > 0).sum())
                A["dev"] += float((e1 - ev0[gi]).sum())
                A["ev0"] += float(ev0[gi].sum())
                A["ev1"] += float(e1.sum())
                A["dmid"] += float((sg * (mid[m][j] - mid[m][gi])
                                    / mid[m][gi] * 1e4).sum())
                # obi1 には欠損があるので nan を伝播させない
                A["dobi"] += float(np.nansum(ob[m][j] - ob[m][gi]))
                A["dofi"] += float(np.nansum(of[m][j] - of[m][gi]))
        # 日次(誤差用)
        row = {"dt": dt}
        for L in LAGS:
            row[f"surv_{int(L*1000)}"] = np.nan
        day_rows.append(row)
        del T, X
        if (k + 1) % 10 == 0:
            print(f"  {k+1}/{len(files)-ntr}", flush=True)

    rows = []
    for L in LAGS:
        A = acc[L]
        n = max(A["n"], 1)
        rows.append({"lag_ms": int(L * 1000), "n": A["n"],
                     "surv": A["surv"] / n,
                     "ev0": A["ev0"] / n, "ev1": A["ev1"] / n,
                     "dev": A["dev"] / n, "dmid": A["dmid"] / n,
                     "dobi": A["dobi"] / n, "dofi": A["dofi"] / n})
    R = pl.DataFrame(rows)
    R.write_csv(DATA / f"decay_{tag}.csv")
    print()
    print("門を通った時点から L 後に、条件がどれだけ残るか(評価期間)")
    print(f"{'L(ms)':>7} {'標本':>10} {'EV>0 の生存率':>13} {'EV(t0)':>9} "
          f"{'EV(t0+L)':>10} {'ΔEV':>9} {'Δmid(bp)':>10} {'ΔOBI':>8}")
    for r in rows:
        print(f"{r['lag_ms']:>7} {r['n']:>10,} {100*r['surv']:>12.1f}% "
              f"{r['ev0']:>9.4f} {r['ev1']:>10.4f} {r['dev']:>9.4f} "
              f"{r['dmid']:>10.4f} {r['dobi']:>8.4f}")
    print(f"\n書き出し {DATA}/decay_{tag}.csv")


if __name__ == "__main__":
    main()
