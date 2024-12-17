export STORE_DIRECTORY=./input

if [ -z "$1" ]; then
    # pass current wd as default directory
    echo "Usage: $0 <ticker>"
    exit 1
fi

TICKER="$1"

python3 scraper/download-kline.py -t spot -s ${TICKER} -i 1mo -y 2024


# unzip all files into /input
echo "Unzipping files into $STORE_DIRECTORY..."

find "$STORE_DIRECTORY/data" -type f -name "*.zip" | while read -r zip_file; do
    echo "Unzipping: $zip_file"
    unzip -o "$zip_file" -d "$STORE_DIRECTORY"
done

# concatenate all csv files into one
COMBINED_FILE="${STORE_DIRECTORY}/${TICKER}_final.csv"

> "$COMBINED_FILE"  # Create or truncate the combined file

find "$STORE_DIRECTORY" -maxdepth 1 -type f -name "*${TICKER}*.csv" | sort | while read -r file; do
    cat "$file" >> "$COMBINED_FILE"
done

# add header to the csv file
python3 src/data_cleaner.py ${COMBINED_FILE}

# clean up the directory
if [ -d "$STORE_DIRECTORY/data" ]; then
    rm -rf "$STORE_DIRECTORY/data"
fi

find "$STORE_DIRECTORY" -maxdepth 1 -type f ! -name "*final*" -exec rm -f {} \;
