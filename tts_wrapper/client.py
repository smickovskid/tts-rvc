import os
import string
import random
from typing import TypedDict
from logging import Logger
import torch
from TTS.api import TTS

from utils.config import check_env_vars
from utils.const import PROJECT_ROOT, UNCONVERTED_DATA_DIR


class TTSConfig(TypedDict):
    tts_model_name: str
    tts_model_dir: str
    tts_config_path: str
    tts_reference_wav: str
    tts_language: str


class TTSWrapper:
    def __init__(self, logger: Logger):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logger
        self.config: TTSConfig = {
            "tts_model_dir": os.getenv("tts_model_dir", ""),
            "tts_config_path": os.getenv("tts_config_path", ""),
            "tts_reference_wav": os.getenv("tts_reference_wav", ""),
            "tts_language": os.getenv("tts_language", ""),
            "tts_model_name": os.getenv("tts_model_name", ""),
        }
        check_env_vars(self.config, self.logger)

        self.tts = TTS(
            model_path=f"{PROJECT_ROOT}/{self.config['tts_model_dir']}",
            config_path=f"{PROJECT_ROOT}/{self.config['tts_config_path']}",
            ).to(self.device)


    def infer_audio(self, text: str) -> str:
        output_file_name = f"{''.join(random.choices(string.ascii_letters, k=7))}.wav"
        self.tts.tts_to_file(
            text=text,
            speaker_wav=f"{PROJECT_ROOT}/{self.config['tts_reference_wav']}",
            language=self.config["tts_language"],
            file_path=f"{UNCONVERTED_DATA_DIR}/{output_file_name}"
        )

        self.logger.info("TTS synthesis complete. Check the 'output.wav' file.")
        return output_file_name
