#!/bin/bash

# Simple script to copy data_imputed_3* files from zion to quetzal
# Source: zion:/ZION/AirPollutionData/Data/MergedDataCSV/16/Imputed/ (remote)
# Destination: /home/olmozavala/DATA/AirPollution/PollutionCSV/ (local quetzal)

ZION_HOST="132.248.8.98"
ZION_PORT="22722"
ZION_USER="olmozavala"
ZION_PATH="/ZION/AirPollutionData/Data/MergedDataCSV/16/Imputed/"
LOCAL_PATH="/home/olmozavala/DATA/AirPollution/PollutionCSV/"

echo "=== Simple CSV Copy Script (Running from quetzal) ==="
echo "From: $ZION_USER@$ZION_HOST:$ZION_PATH"
echo "To: $LOCAL_PATH (local quetzal)"
echo ""

# Copy newer files from zion to local quetzal
echo "Copying newer data_imputed_3* files from zion to quetzal..."
rsync -avz --progress --include="data_imputed_7*" --exclude="*" \
    -e "ssh -p $ZION_PORT" \
    "$ZION_USER@$ZION_HOST:$ZION_PATH/" \
    "$LOCAL_PATH"

if [ $? -eq 0 ]; then
    echo "Copy completed successfully!"
else
    echo "Copy failed!"
    exit 1
fi
