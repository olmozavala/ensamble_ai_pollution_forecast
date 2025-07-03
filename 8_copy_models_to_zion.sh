#!/bin/bash

# Script to copy model files from skynet to zion
# Usage: ./8_copy_models_to_zion.sh <model_name>
# Example: ./8_copy_models_to_zion.sh MeanPollutantsNewCSVFilesImputedWithClimatologyBootstrap3

# Check if model name is provided
if [ $# -eq 0 ]; then
    echo "Error: Model name is required"
    echo "Usage: $0 <model_name>"
    echo "Example: $0 MeanPollutantsNewCSVFilesImputedWithClimatologyBootstrap3"
    exit 1
fi
# Parallel_all_prev24_heads4_w4_p4_ar8_bootstrapTrue_thresh2_weather4_2/0701_101128

MODEL_NAME="$1"

# Zion connection details
ZION_HOST="132.248.8.98"
ZION_PORT="22722"
ZION_USER="olmozavala"

# Source paths on skynet (local)
SKYNET_MODELS_PATH="/unity/f1/ozavala/DATA/AirPollution/OUTPUT/models"
SKYNET_PREDICTIONS_PATH="/unity/f1/ozavala/DATA/AirPollution/OUTPUT/predictions"
SKYNET_CONFIGS_PATH="/unity/f1/ozavala/CODE/ensamble_ai_pollution_forecast/saved_confs/parallel_configs"

# Destination paths on zion (remote)
ZION_BASE_PATH="/ZION/AirPollutionData/pedro_files"
ZION_MODELS_PATH="$ZION_BASE_PATH/models"
ZION_PREDICTIONS_PATH="$ZION_BASE_PATH/predictions"
ZION_CONFIGS_PATH="$ZION_BASE_PATH/configs"

echo "=== Model Copy Script (Running from skynet) ==="
echo "Model: $MODEL_NAME"
echo "From: skynet (local)"
echo "To: $ZION_USER@$ZION_HOST:$ZION_BASE_PATH"
echo ""

# Create destination directories on zion if they don't exist
echo "Creating destination directories on zion..."
ssh -p $ZION_PORT $ZION_USER@$ZION_HOST "mkdir -p $ZION_MODELS_PATH $ZION_PREDICTIONS_PATH $ZION_CONFIGS_PATH"

if [ $? -ne 0 ]; then
    echo "Failed to create directories on zion!"
    exit 1
fi

# Copy model weights (only best model)
echo "Copying best model weights..."
if [ -d "$SKYNET_MODELS_PATH/$MODEL_NAME" ]; then
    # Find the best model checkpoint (usually the one with highest epoch number or 'best' in name)
    BEST_MODEL=$(find "$SKYNET_MODELS_PATH/$MODEL_NAME" -name "*.pth" | sort -V | tail -1)
    
    if [ -n "$BEST_MODEL" ]; then
        echo "Found best model: $(basename "$BEST_MODEL")"
        
        # Create destination directory
        ssh -p $ZION_PORT $ZION_USER@$ZION_HOST "mkdir -p $ZION_MODELS_PATH/$MODEL_NAME"
        
        # Copy only the best model
        rsync -avz --progress \
            -e "ssh -p $ZION_PORT" \
            "$BEST_MODEL" \
            "$ZION_USER@$ZION_HOST:$ZION_MODELS_PATH/$MODEL_NAME/"
        
        if [ $? -eq 0 ]; then
            echo "Best model weights copied successfully!"
        else
            echo "Failed to copy best model weights!"
            exit 1
        fi
    else
        echo "Warning: No .pth model files found in: $SKYNET_MODELS_PATH/$MODEL_NAME"
    fi
else
    echo "Warning: Model weights directory not found: $SKYNET_MODELS_PATH/$MODEL_NAME"
fi

# Copy predictions
echo "Copying model predictions..."
if [ -d "$SKYNET_PREDICTIONS_PATH/$MODEL_NAME" ]; then
    rsync -avz --progress \
        -e "ssh -p $ZION_PORT" \
        "$SKYNET_PREDICTIONS_PATH/$MODEL_NAME/" \
        "$ZION_USER@$ZION_HOST:$ZION_PREDICTIONS_PATH/$MODEL_NAME/"
    
    if [ $? -eq 0 ]; then
        echo "Model predictions copied successfully!"
    else
        echo "Failed to copy model predictions!"
        exit 1
    fi
else
    echo "Warning: Model predictions directory not found: $SKYNET_PREDICTIONS_PATH/$MODEL_NAME"
fi

# Copy config files
echo "Copying config files..."
if [ -d "$SKYNET_CONFIGS_PATH" ]; then
    rsync -avz --progress \
        -e "ssh -p $ZION_PORT" \
        "$SKYNET_CONFIGS_PATH/" \
        "$ZION_USER@$ZION_HOST:$ZION_CONFIGS_PATH/"
    
    if [ $? -eq 0 ]; then
        echo "Config files copied successfully!"
    else
        echo "Failed to copy config files!"
        exit 1
    fi
else
    echo "Warning: Config files directory not found: $SKYNET_CONFIGS_PATH"
fi

echo ""
echo "=== Copy completed successfully! ==="
echo "Model: $MODEL_NAME"
echo "Files copied to zion at: $ZION_BASE_PATH"
