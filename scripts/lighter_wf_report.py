r"""ウォークフォワードの集計 — 日単位ブートストラップ・戦略数の記録・凍結候補の選定。

=============================================================================
出す統計(要求 4・5)
=============================================================================
(A) 選択つきウォークフォワード
    fold ごとに **検証期間の損益が最大の構成** を選び、その構成の **テスト日** の
    損益だけを繋ぐ。実運用のパイプラインに対応する唯一の推定量。

(B) 固定構成の格子(全 63 通り)
    各構成のテスト日損益を全 fold で足す。**試した戦略数の台帳**そのもの。
    最良だけを見ないために全件を出し、Bonferroni 閾値を併記する。

(C) 日単位ブートストラップ
    **同じ日の全銘柄をまとめて** 復元抽出する。市場共通ショックによる
    銘柄間の依存を壊さない。統計量は「1 取引あたり損益」と「日次総額」の 2 つ。
    ★日をまたいで独立を仮定しているだけで、日内の依存は残している。

★これらはすべて **探索的**(私が既に見た 08-18〜09-05 の期間)。
  確定的な判定は `lighter_oos.py` の未見期間(09-06〜)で行う。

出力: E:/Memory-lighter/wf/wf_summary.parquet, wf_boot.parquet
      config/lighter_frozen.json(凍結候補。人間が確認してから使う)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import norm

SRC = Path("E:/Memory-lighter/wf")
ROOT = Path(__file__).resolve().parent.parent
NBOOT = 5000
EXPLORE_END = "2026-09-05"


def boot_days(df: pl.DataFrame, rng, nboot=NBOOT):
    """日を単位に復元抽出(同じ日の全銘柄をまとめて)。"""
    days = df["day"].unique().to_list()
    by = {d: df.filter(pl.col("day") == d) for d in days}
    tot = np.array([by[d]["pnl_sum"].sum() for d in days])
    cnt = np.array([by[d]["n_trade"].sum() for d in days])
    gro = np.array([by[d]["gross_sum"].sum() for d in days])
    cst = np.array([by[d]["cost_sum"].sum() for d in days])
    nd = len(days)
    if nd < 3:
        return {}
    idx = rng.integers(0, nd, size=(nboot, nd))
    bt = tot[idx].sum(axis=1)
    bc = cnt[idx].sum(axis=1)
    per = np.where(bc > 0, bt / np.maximum(bc, 1), np.nan)
    return {
        "n_day": nd, "n_trade": int(cnt.sum()), "pnl_sum": float(tot.sum()),
        "pnl_per_trade": float(tot.sum() / max(cnt.sum(), 1)),
        "day_pos": int((tot > 0).sum()),
        # ★粗利と費用を分ける(「信号に価値が無い」のか「費用が高い」のかは別問題)
        "gross_per_trade": float(gro.sum() / max(cnt.sum(), 1)),
        "cost_per_trade": float(cst.sum() / max(cnt.sum(), 1)),
        "boot_sum_lo": float(np.nanpercentile(bt, 2.5)),
        "boot_sum_hi": float(np.nanpercentile(bt, 97.5)),
        "boot_per_lo": float(np.nanpercentile(per, 2.5)),
        "boot_per_hi": float(np.nanpercentile(per, 97.5)),
        "boot_p_gt0": float(np.mean(bt > 0)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    a = ap.parse_args()
    rng = np.random.default_rng(20260908)
    import glob
    fs = sorted(glob.glob(str(SRC / "ledger_*.parquet")))
    if not fs:
        raise SystemExit("ledger_*.parquet が無い")
    L = pl.concat([pl.read_parquet(f) for f in fs], how="diagonal_relaxed")
    L = L.filter(pl.col("day") <= EXPLORE_END)
    ncfg = L.select(["signal", "q", "h"]).unique().height
    print(f"台帳 {L.height:,} 行 / 構成 {ncfg} 通り / "
          f"銘柄 {L['symbol'].n_unique()} / fold {L['fold'].n_unique()}")

    # ---------- (B) 固定構成の格子 ----------
    rows = []
    te = L.filter(pl.col("split") == "test")
    for (sg, q, h), g in te.group_by(["signal", "q", "h"]):
        st = boot_days(g, rng)
        if st:
            rows.append({"signal": sg, "q": q, "h": h, **st})
    G = pl.DataFrame(rows).sort("pnl_sum", descending=True)
    G.write_parquet(SRC / "wf_summary.parquet")
    zb = norm.isf(0.025 / max(ncfg, 1))
    print(f"\n=== (B) 固定構成 {len(rows)} 通りのテスト日集計 ===")
    print(f"Bonferroni(両側 0.05 / {ncfg})の z 閾値 = {zb:.2f}")
    print(G.head(12))
    print("...")
    print(G.tail(5))
    npos = int((G["pnl_sum"] > 0).sum())
    ngro = int((G["gross_per_trade"] > 0).sum())
    gm = float(G["gross_per_trade"].median())
    cm = float(G["cost_per_trade"].median())
    print(f"\n純損益の総額が正の構成: {npos} / {len(rows)}")
    print(f"★費用を引く前(粗利)が正の構成: {ngro} / {len(rows)}")
    print(f"  粗利/取引 の中央値 {gm:+.4f} bp / 費用/取引 の中央値 {cm:+.4f} bp")
    print(f"  → 費用は粗利の {cm / max(abs(gm), 1e-9):.0f} 倍。"
          f"損益分岐スプレッドは {gm:.4f} bp(往復)")
    strong = G.filter(pl.col("boot_p_gt0") > 0.975)
    print(f"日ブートストラップで 97.5% 以上が正: {strong.height} / {len(rows)}")

    # ---------- (A) 選択つきウォークフォワード ----------
    sel = []
    va = L.filter(pl.col("split") == "val")
    for (sym, fold), gv in va.group_by(["symbol", "fold"]):
        agg = gv.group_by(["signal", "q", "h"]).agg(
            pl.col("pnl_sum").sum().alias("v"))
        if agg.height == 0:
            continue
        best = agg.sort("v", descending=True).row(0, named=True)
        gt = te.filter((pl.col("symbol") == sym) & (pl.col("fold") == fold)
                       & (pl.col("signal") == best["signal"])
                       & (pl.col("q") == best["q"]) & (pl.col("h") == best["h"]))
        for r in gt.iter_rows(named=True):
            sel.append({**r, "val_pnl": best["v"]})
    S = pl.DataFrame(sel)
    S.write_parquet(SRC / "wf_selected.parquet")
    st = boot_days(S, rng)
    print("\n=== (A) 検証で選び、テストで評価(実運用のパイプライン)===")
    for k, v in st.items():
        print(f"  {k}: {v:,.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("\n選ばれた構成の頻度:")
    print(S.group_by(["signal", "q", "h"]).len().sort("len", descending=True).head(8))

    # ---------- (A2) 全銘柄共通で 1 構成だけ選ぶ(凍結規則と同型)----------
    # ★(A) は銘柄 × fold ごとに 63 通りから選ぶので選択の自由度が 144 倍ある。
    #   凍結して未見期間で試すのは「全銘柄共通の 1 構成」なので、
    #   探索期間でも同じ形の推定量を出しておかないと比較にならない。
    sel2 = []
    for fold in sorted(L["fold"].unique().to_list()):
        gv = va.filter(pl.col("fold") == fold)
        if gv.height == 0:
            continue
        agg = gv.group_by(["signal", "q", "h"]).agg(
            pl.col("pnl_sum").sum().alias("v")).sort("v", descending=True)
        b = agg.row(0, named=True)
        gt = te.filter((pl.col("fold") == fold) & (pl.col("signal") == b["signal"])
                       & (pl.col("q") == b["q"]) & (pl.col("h") == b["h"]))
        for r in gt.iter_rows(named=True):
            sel2.append(r)
    S2 = pl.DataFrame(sel2)
    S2.write_parquet(SRC / "wf_selected_global.parquet")
    st2 = boot_days(S2, rng)
    print("\n=== (A2) fold ごとに全銘柄共通の 1 構成を選ぶ(凍結規則と同型)===")
    for k, v in st2.items():
        print(f"  {k}: {v:,.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ---------- 凍結候補 ----------
    # ★検証期間の成績だけで決める(テスト期間は見ない)。
    vv = []
    for (sg, q, h), g in va.group_by(["signal", "q", "h"]):
        vv.append({"signal": sg, "q": q, "h": h,
                   "val_sum": float(g["pnl_sum"].sum()),
                   "val_n": int(g["n_trade"].sum())})
    V = pl.DataFrame(vv).sort("val_sum", descending=True)
    print("\n=== 凍結候補(検証期間の総額だけで決定。テストは見ていない)===")
    print(V.head(6))
    top = V.row(0, named=True)
    frozen = {
        "created": "2026-09-08",
        "purpose": "未見期間(2026-09-06 以降)での確定的検定に使う。以後変更しない。",
        "model": top["signal"],
        "threshold_rule": {
            "type": "trailing_quantile_of_abs_signal",
            "q": top["q"],
            "window_days": 7,
            "note": "テスト日の直前 7 日で毎回引き直す。テスト日は一切見ない",
        },
        "sign_rule": "直前 7 日で signal と 60 秒後リターンの相関の符号を取る",
        "size": {"type": "fixed", "unit": 1.0,
                 "note": "全取引で同じ。単価と総額の両方を報告する"},
        "exit": {"type": "time", "seconds": top["h"]},
        "cost": {"spread_bp": "その時点の実測スプレッドを往復で 1 回引く",
                 "fee_bp": 0.0,
                 "note": "Lighter の料率はデータから確定できないため 0。楽観側"},
        "position": "1 銘柄 1 ポジション(保有中は新規建てしない)",
        "purge_seconds": 300,
        "symbols": "H100 を除く 12 銘柄(板の被覆 14.7% のため事前除外)",
        "n_configs_searched": ncfg,
        "bonferroni_z": round(float(zb), 3),
        "selected_on": "検証期間の総額(探索期間 08-18〜09-05 の内側)",
    }
    (ROOT / "config").mkdir(exist_ok=True)
    with open(ROOT / "config" / "lighter_frozen.json", "w", encoding="utf-8") as f:
        json.dump(frozen, f, ensure_ascii=False, indent=2)
    print(f"\n凍結 → config/lighter_frozen.json: {top['signal']} "
          f"q={top['q']} h={top['h']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
