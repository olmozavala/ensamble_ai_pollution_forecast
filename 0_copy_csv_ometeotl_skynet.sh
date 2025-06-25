#!/bin/bash

# Simple script to copy data_imputed_3* files from zion to quetzal
# Source: zion:/ZION/AirPollutionData/Data/MergedDataCSV/16/Imputed/ (remote)
# Destination: /home/olmozavala/DATA/AirPollution/PollutionCSV/ (local quetzal)

OMETEOTL_HOST="ometeotl.atmosfera.unam.mx"
OMETEOTL_PORT="9022"
OMETEOTL_USER="olmozavala"
OMETEOTL_PATH="/home/olmozavala/DATA/AirPollution/PollutionCSV"
LOCAL_PATH="/unity/f1/ozavala/DATA/AirPollution/PollutionCSV"

echo "=== Simple CSV Copy Script (Running from skynet) ==="
echo "From: $OMETEOTL_USER@$OMETEOTL_HOST:$OMETEOTL_PATH"
echo "To: $LOCAL_PATH (local skynet)"
echo ""

# Copy newer files from ometeotl to local skynet
echo "Copying newer data_imputed_7* files from ometeotl to skynet..."
rsync -avz --progress --include="data_imputed_7*" --exclude="*" \
    -e "ssh -p $OMETEOTL_PORT" \
    "$OMETEOTL_USER@$OMETEOTL_HOST:$OMETEOTL_PATH/" \
    "$LOCAL_PATH"

if [ $? -eq 0 ]; then
    echo "Copy completed successfully!"
else
    echo "Copy failed!"
    exit 1
fi


# Define paths for weather files
OMETEOTL_WRF_PATH="/home/olmozavala/DATA/AirPollution/WRF_NetCDF"
LOCAL_WRF_PATH="/unity/f1/ozavala/DATA/AirPollution/WRF_NetCDF"

echo ""
echo "=== Copying Weather Files ==="
echo "From: $OMETEOTL_USER@$OMETEOTL_HOST:$OMETEOTL_WRF_PATH"
echo "To: $LOCAL_WRF_PATH (local skynet)"
echo ""

# Copy weather files from ometeotl to local skynet
echo "Copying weather files from ometeotl to skynet..."
rsync -avz --progress \
    -e "ssh -p $OMETEOTL_PORT" \
    "$OMETEOTL_USER@$OMETEOTL_HOST:$OMETEOTL_WRF_PATH/" \
    "$LOCAL_WRF_PATH"

if [ $? -eq 0 ]; then
    echo "Weather files copy completed successfully!"
else
    echo "Weather files copy failed!"
    exit 1
fi
