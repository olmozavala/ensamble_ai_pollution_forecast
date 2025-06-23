#!/bin/bash

# Simple script to copy data_imputed_3* files to remote server
# Source: /home/olmozavala/DATA/AirPollution/PollutionCSV (local)
# Destination: ometeotl.atmosfera.unam.mx:/home/olmozavala/DATA/AirPollution/PollutionCSV/

REMOTE_HOST="ometeotl.atmosfera.unam.mx"
REMOTE_PORT="9022"
REMOTE_USER="olmozavala"
REMOTE_PATH="/home/olmozavala/DATA/AirPollution/PollutionCSV/"
LOCAL_PATH="/ZION/AirPollutionData/Data/MergedDataCSV/16/Imputed/"

echo "=== Simple CSV Copy Script ==="
echo "From: $LOCAL_PATH (local)"
echo "To: $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
echo ""

# Copy newer files to remote server
echo "Copying newer data_imputed_3* files to remote server..."
rsync -avz --progress --include="data_imputed_3*" --exclude="*" \
    -e "ssh -p $REMOTE_PORT" \
    "$LOCAL_PATH/" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"

if [ $? -eq 0 ]; then
    echo "Copy completed successfully!"
else
    echo "Copy failed!"
    exit 1
fi
