#!/bin/bash

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <symbol> <interval> <start_year> [month] [day]"
    exit 1
fi

symbol=$1
interval=$2
year=$3

# default to 1
month=${4:-1}
day=${5:-1}

python3 src/get_data.py ${symbol} ${interval} ${year} ${month} ${day}