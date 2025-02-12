#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <results_dir>"
fi

FILE="$1"

python3 ../src/backtest_metrics.py ${FILE}