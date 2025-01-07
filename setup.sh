#!/bin/bash
set -euxo pipefail

# Default values for flags
COQUI_MODEL_ZIP=""
RVC_MODEL_ZIP=""
NAME=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --coqui_model_zip)
      COQUI_MODEL_ZIP="$2"
      shift 2
      ;;
    --rvc_model_zip)
      RVC_MODEL_ZIP="$2"
      shift 2
      ;;
    --name)
      NAME="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Ensure required arguments are provided
if [[ -z "$COQUI_MODEL_ZIP" || -z "$RVC_MODEL_ZIP" || -z "$NAME" ]]; then
  echo "Error: Missing required arguments. Usage:"
  echo "./script.sh --coqui_model_zip <url> --rvc_model_zip <url> --name <name>"
  exit 1
fi

pip install --upgrade pip==24
pip install -r requirements.txt

# Create necessary directories
mkdir -p data data/converted data/unconverted
mkdir -p "tts_models/$NAME" "tts_models/$NAME/reference_files"
mkdir -p "assets/weights"

# Check if models are already downloaded
if [ ! -f assets/.downloaded ]; then
  curl -o tmp/download_models.py https://raw.githubusercontent.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/refs/heads/main/tools/download_models.py
  python tmp/download_models.py
  touch assets/.downloaded
else
  echo "Models already downloaded. Skipping download."
fi


# Download and extract Coqui model
if [[ -n "$COQUI_MODEL_ZIP" ]]; then
  curl -o "tmp/coqui_model.zip" "$COQUI_MODEL_ZIP"
  unzip -o "tmp/coqui_model.zip" -d "tts_models/$NAME"
fi

# Download and extract RVC model
if [[ -n "$RVC_MODEL_ZIP" ]]; then
  curl -o "tmp/rvc_model.zip" "$RVC_MODEL_ZIP"
  unzip -o "tmp/rvc_model.zip" -d "assets/weights"
fi

# Cleanup
rm -rf tmp
