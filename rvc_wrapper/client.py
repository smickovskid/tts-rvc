from logging import Logger
import os
from typing import TypedDict
from pathlib import Path
from scipy.io import wavfile
from rvc.modules.vc.modules import VC

from utils.config import check_env_vars
from utils.const import CONVERTED_DATA_DIR, PROJECT_ROOT, UNCONVERTED_DATA_DIR


class RVCConfig(TypedDict):
    hubert_path: str
    model_path: str
    index_root: str
    rmvpe_root: str


class RVCWrapper:
    def __init__(self, logger: Logger):
        self.logger = logger
        self.index_file = os.getenv("index_file")
        self.config: RVCConfig = {
            "hubert_path": os.getenv("hubert_path", ""),
            "model_path": os.getenv("model_path", ""),
            "index_root": os.getenv("index_root", ""),
            "rmvpe_root": os.getenv("rmvpe_root", ""),
        }

        check_env_vars(dict(self.config), self.logger)

        self.vc = VC()
        self.vc.get_vc(f"{PROJECT_ROOT}/{self.config['model_path']}")

    def infer_audio(self, ref_audio: str):
        index_path = None
        if self.index_file:
            index_path = Path(f"{PROJECT_ROOT}/{self.index_file}")

        tgt_sr, audio_opt, _, _ = self.vc.vc_single(
            1,
            input_audio_path=Path(ref_audio),
            hubert_path=(f"{PROJECT_ROOT}/{self.config['hubert_path']}"),
            index_file=index_path,
        )
        filename = os.path.basename(ref_audio)
        wavfile.write(f"{CONVERTED_DATA_DIR}/converted_{filename}", tgt_sr, audio_opt)
        return f"{CONVERTED_DATA_DIR}/converted_{filename}"
