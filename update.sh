#!/bin/bash


if [ "$#" -ne 1 ]; then
	echo "Usage: $0 <path_to_data>"
	exit 1
fi

python3 src/update_data.py $1
