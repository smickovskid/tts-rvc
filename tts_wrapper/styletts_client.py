
import os
import string
import random
from typing import TypedDict
from logging import Logger
from TTS.tts.models.xtts import Path
import torch
from styletts2 import tts
from utils.config import check_env_vars
from utils.const import PROJECT_ROOT, UNCONVERTED_DATA_DIR


class StyleTTSConfig(TypedDict):
    styletts_model_dir: str

class StyleTTSWrapper:
    def __init__(self, logger: Logger):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logger
        self.config: StyleTTSConfig = {
            "styletts_model_dir": os.getenv("styletts_model_dir", ""),
        }
        check_env_vars(dict(self.config), self.logger)

        self.styletts_config_path = f"{PROJECT_ROOT}/{self.config['styletts_model_dir']}/config.yml"
        self.styletts_model_path = f"{PROJECT_ROOT}/{self.config['styletts_model_dir']}/model.pth"
        self.reference_file = Path(f"{PROJECT_ROOT}/{self.config['styletts_model_dir']}/reference.wav")

        for path in [self.styletts_config_path, self.styletts_model_path, self.reference_file]:
            if not Path(path).exists():
                raise FileNotFoundError(f"Required file does not exist: {path}")

        self.tts = tts.StyleTTS2(model_checkpoint_path=self.styletts_model_path, config_path=self.styletts_config_path)

    def infer_audio(self, text: str) -> str:
        output_file_name = f"{''.join(random.choices(string.ascii_letters, k=7))}.wav"
        output_file_path = f"{UNCONVERTED_DATA_DIR}/{output_file_name}"
        self.tts.inference(text, target_voice_path=self.reference_file, output_wav_file=output_file_path, diffusion_steps=10)

        return output_file_path




