from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# TTS
TTS_MODELS_DIR = PROJECT_ROOT / "tts_models"

# RVC
ASSETS_DIR = PROJECT_ROOT / "assets"

# Data conversion
UNCONVERTED_DATA_DIR = PROJECT_ROOT / "data" / "unconverted"
CONVERTED_DATA_DIR = PROJECT_ROOT / "data" / "converted"

