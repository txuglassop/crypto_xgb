#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <path_to_data>"
    exit 1
fi

FILE="$1"

python3 src/check_data.py ${FILE}