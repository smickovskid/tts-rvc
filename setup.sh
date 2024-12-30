#!/bin/bash
set -euxo pipefail

pip install -r requirements.txt

mkdir -p tmp
curl https://raw.githubusercontent.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/refs/heads/main/tools/download_models.py > tmp/download_models.py
python tmp/download_models.py
