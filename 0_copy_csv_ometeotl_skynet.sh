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

# Function to ask for confirmation
ask_confirmation() {
    local message="$1"
    echo ""
    read -p "$message (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping this section."
        return 1
    fi
    return 0
}

# Finally copy normalization parameter files
OMETEOTL_NORM_PATH="/home/olmozavala/DATA/AirPollution/TrainingData"
LOCAL_NORM_PATH="/unity/f1/ozavala/DATA/AirPollution/TrainingData"

echo ""
echo "=== Copying Normalization Parameter Files ==="
echo "From: $OMETEOTL_USER@$OMETEOTL_HOST:$OMETEOTL_NORM_PATH"
echo "To: $LOCAL_NORM_PATH (local skynet)"
echo ""

# Ask before copying norm_params files
if ask_confirmation "Do you want to copy normalization parameter files (*.yml)?"; then
    # Copy norm_params files from ometeotl to local skynet
    echo "Copying norm_params files from ometeotl to skynet..."
    rsync -avz --progress --include="*.yml" --exclude="*" \
        -e "ssh -p $OMETEOTL_PORT" \
        "$OMETEOTL_USER@$OMETEOTL_HOST:$OMETEOTL_NORM_PATH/" \
        "$LOCAL_NORM_PATH"

    if [ $? -eq 0 ]; then
        echo "Normalization parameter files copy completed successfully!"
    else
        echo "Normalization parameter files copy failed!"
        exit 1
    fi
fi

# Ask before copying training data files
if ask_confirmation "Do you want to copy training data files (*.pkl)?"; then
    # Copy training data files from ometeotl to local skynet
    echo "Copying training data files from ometeotl to skynet..."
    rsync -avz --progress --include="*.pkl" --exclude="*" \
        -e "ssh -p $OMETEOTL_PORT" \
        "$OMETEOTL_USER@$OMETEOTL_HOST:$OMETEOTL_NORM_PATH/" \
        "$LOCAL_NORM_PATH"

    if [ $? -eq 0 ]; then
        echo "Training data files copy completed successfully!"
    else
        echo "Training data files copy failed!"
        exit 1
    fi
fi

# Ask before copying pollution data files
if ask_confirmation "Do you want to copy pollution data files (data_imputed_7*)?"; then
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
fi

# Define paths for weather files
OMETEOTL_WRF_PATH="/home/olmozavala/DATA/AirPollution/WRF_NetCDF"
LOCAL_WRF_PATH="/unity/f1/ozavala/DATA/AirPollution/WRF_NetCDF"

echo ""
echo "=== Copying Weather Files ==="
echo "From: $OMETEOTL_USER@$OMETEOTL_HOST:$OMETEOTL_WRF_PATH"
echo "To: $LOCAL_WRF_PATH (local skynet)"
echo ""

# Ask before copying weather files
if ask_confirmation "Do you want to copy weather files (WRF_NetCDF)?"; then
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
fi

echo ""
echo "Script completed!"

