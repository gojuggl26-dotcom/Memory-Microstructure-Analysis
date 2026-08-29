"""検定統計量の参照分布を明示し直す。

- プール推定(n = 2,345 万)の Wald 統計量は漸近正規 → z と呼ぶ
- 日次クラスター(G = 99)と Fama-MacBeth(99 日)は自由度 98 の t 分布を使う
  併せてクラスター頑健分散の小標本補正 c = G/(G-1) · (N-1)/(N-K) を入れる
時刻添字の t とは別物なので、レポートでは統計量を z / t_98 と表記する。
"""
from __future__ import annotations
import json, math
from pathlib import Path

OUT = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")


def betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < tiny:
            d = tiny
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < tiny:
            d = tiny
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 3e-16:
            break
    return h


def betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lb = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
          + a * math.log(x) + b * math.log1p(-x))
    bt = math.exp(lb)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * betacf(a, b, x) / a
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b


def p_t(stat: float, df: int) -> float:
    return betai(df / 2.0, 0.5, df / (df + stat * stat))


def p_z(stat: float) -> float:
    return math.erfc(abs(stat) / math.sqrt(2.0))


def main() -> None:
    G = 99
    r = json.loads((OUT / "regression.json").read_text(encoding="utf-8"))
    # クラスター頑健分散の小標本補正(K=2)
    out = {"reference": "pooled: z ~ N(0,1) / day-level: t(98), cluster ssc = G/(G-1)*(N-1)/(N-K)"}
    rows = []
    for k in ["event_1", "event_5", "event_20", "clock_1", "clock_5", "clock_20"]:
        s = r["specs"][k]
        n = s["n"]
        ssc = math.sqrt(G / (G - 1) * (n - 1) / (n - 2))
        se_c = s["se_cluster_day"] * ssc
        z_nw = s["beta"] / s["se_newey_west"]
        t_c = s["beta"] / se_c
        f = r["fama_macbeth"][k]
        t_fm = f["beta_mean"] / f["se"]
        rows.append({"spec": k, "beta": s["beta"],
                     "z_nw": z_nw, "p_z_nw": p_z(z_nw),
                     "t98_cluster": t_c, "p_t98_cluster": p_t(t_c, G - 1),
                     "se_cluster_ssc": se_c,
                     "t98_fm_iid": t_fm, "p_t98_fm_iid": p_t(t_fm, G - 1),
                     "p_normal_fm_iid_old": p_z(t_fm)})
    out["specs"] = rows
    a = json.loads((OUT / "audit.json").read_text(encoding="utf-8"))
    t1 = a["T1_fama_macbeth"]
    out["audit_T1"] = {
        "t98_iid": t1["t_iid"], "p_t98_iid": p_t(t1["t_iid"], G - 1),
        "t98_nw_days_lag5": t1["t_newey_west_days"],
        "p_t98_nw_days_lag5": p_t(t1["t_newey_west_days"], G - 1),
    }
    (OUT / "notation_fix.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    for x in rows:
        print(f"{x['spec']:>9} beta={x['beta']:+.4f} z_NW={x['z_nw']:+7.2f} (p={x['p_z_nw']:.2e}) "
              f"t98_clus={x['t98_cluster']:+6.2f} (p={x['p_t98_cluster']:.2e}) "
              f"t98_FM={x['t98_fm_iid']:+6.2f} (p={x['p_t98_fm_iid']:.2e} / 旧正規 {x['p_normal_fm_iid_old']:.2e})")
    print(json.dumps(out["audit_T1"], indent=1))


if __name__ == "__main__":
    main()
