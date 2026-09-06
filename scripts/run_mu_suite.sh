#!/bin/sh
# MU で作った分析一式を、指定した銘柄でそのまま走らせる。
#
#   sh scripts/run_mu_suite.sh xyz:AMD
#
# 入力は data/bbo_<tag>.parquet と data/fills_<tag>.parquet だけ。
# どちらも E:/hlpipe のミラーから scripts/mirror_to_local.py で作れるので
# S3 の egress は発生しない。
#
# 実行順の制約:
#   build_oi_volume が oi_series_<tag>.parquet を書き、build_variance が読む。
#   それ以外は独立。
#
# 失敗したステップは記録して次へ進む(銘柄によっては約定が薄くて
# 当てはめが立たないものがある)。最後に失敗一覧を出す。

set -u
cd /c/Users/ii562/Downloads/Memory
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
COIN="$1"
TAG=$(echo "$COIN" | tr ':' '_')
FAILED=""

run() {
  name="$1"; shift
  printf "  [%s] %s ... " "$(date +%H:%M:%S)" "$name"
  if "$@" > "data/_suite_${TAG}_${name}.log" 2>&1; then
    echo "ok"
  else
    echo "NG"
    FAILED="$FAILED $name"
  fi
}

echo "===== $COIN のスイート開始 $(date +%H:%M:%S) ====="

# --- A. この銘柄はどういう市場か -------------------------------------------
run oi_volume      $PY scripts/build_oi_volume.py  --coin "$COIN"
run volume_side    $PY scripts/build_volume_side.py --coin "$COIN"
run order_size     $PY scripts/build_order_size.py --coin "$COIN"
run vol            $PY scripts/build_vol.py        --coin "$COIN"
run variance       $PY scripts/build_variance.py   --coin "$COIN"
run sign_chain     $PY scripts/build_sign_chain.py --coin "$COIN"

# --- B. 板の状態は将来の値動きを教えてくれるか -----------------------------
run microprice     $PY scripts/build_microprice.py --coin "$COIN"
run obi_ofi        $PY scripts/build_obi_ofi.py    --coin "$COIN"
run book_slope     $PY scripts/build_book_slope.py --coin "$COIN"
run cancel_rate    $PY scripts/build_cancel_rate.py --coin "$COIN"
run depth          $PY scripts/build_depth.py      --coin "$COIN"

# --- C. 注文フローはどれだけ自分自身を引きずるか ---------------------------
run sign_persist   $PY scripts/build_sign_persistence.py --coin "$COIN"
run acf_200ms      $PY scripts/build_acf_200ms.py  --coin "$COIN"
run var100         $PY scripts/build_var100.py     --coin "$COIN"

# --- D. 板の中の量どうしはどう結びついているか -----------------------------
run spread_flow    $PY scripts/build_spread_flow.py --coin "$COIN"
run markout        $PY scripts/build_markout.py    --coin "$COIN"
run resilience     $PY scripts/build_resilience.py --coin "$COIN"

# --- 図 ---------------------------------------------------------------------
for P in oi_volume volume_side order_size vol microprice obi_ofi book_slope \
         cancel_rate depth sign_persistence acf_200ms var100 spread_flow \
         markout resilience price; do
  run "plot_$P" $PY "scripts/plot_$P.py" --coin "$COIN"
done
run variance_chart $PY scripts/variance_chart.py --coin "$COIN"

echo "===== $COIN 終了 $(date +%H:%M:%S) ====="
if [ -n "$FAILED" ]; then
  echo "FAILED[$COIN]:$FAILED"
else
  echo "FAILED[$COIN]: なし"
fi
