export STORE_DIRECTORY=./input

if [ $# -ne 3 ]; then
    echo "Usage: $0 <ticker> <WINDOW> <YEAR>"
    exit 1
fi

TICKER="$1"
WINDOW="$2"
YEAR="$3"

python3 scraper/download-kline.py -t spot -s ${TICKER} -i ${WINDOW} -y ${YEAR}


# unzip all files into /input
echo "Unzipping files into $STORE_DIRECTORY..."

find "$STORE_DIRECTORY/data" -type f -name "*.zip" | while read -r zip_file; do
    unzip -o "$zip_file" -d "$STORE_DIRECTORY"
done

# concatenate all csv files into one
COMBINED_FILE="${STORE_DIRECTORY}/${TICKER}_${WINDOW}_${YEAR}_final.csv"

> "$COMBINED_FILE"  # Create or truncate the combined file

find "$STORE_DIRECTORY" -maxdepth 1 -type f -name "*${TICKER}*.csv" ! -name "*final*.csv" | sort | while read -r file; do
    cat "$file" >> "$COMBINED_FILE"
done

# add header to the csv file
python3 src/data_cleaner.py ${COMBINED_FILE}

# clean up the directory
if [ -d "$STORE_DIRECTORY/data" ]; then
    rm -rf "$STORE_DIRECTORY/data"
fi

find "$STORE_DIRECTORY" -maxdepth 1 -type f ! -name "*final*" -exec rm -f {} \;
