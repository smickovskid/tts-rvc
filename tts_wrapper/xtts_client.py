import os
import string
import random
from typing import TypedDict
from logging import Logger
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts, torchaudio
from utils.config import check_env_vars
from utils.const import PROJECT_ROOT, UNCONVERTED_DATA_DIR


class XTTSV2Config(TypedDict):
    tts_model_dir: str
    tts_config_path: str
    tts_vocab_path: str
    tts_reference_wav: str
    tts_language: str
    tts_speakers: str


class XTTSWrapper:
    def __init__(self, logger: Logger):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logger
        self.config: XTTSV2Config = {
            "tts_model_dir": os.getenv("tts_model_dir", ""),
            "tts_config_path": os.getenv("tts_config_path", ""),
            "tts_vocab_path": os.getenv("tts_vocab_path", ""),
            "tts_reference_wav": os.getenv("tts_reference_wav", ""),
            "tts_speakers": os.getenv("tts_speakers", ""),
            "tts_language": os.getenv("tts_language", ""),
        }
        check_env_vars(self.config, self.logger)

        # Load XTTS configuration and model
        xtts_config_path = f"{PROJECT_ROOT}/{self.config['tts_config_path']}"
        xtts_model_path = f"{PROJECT_ROOT}/{self.config['tts_model_dir']}/model.pth"
        xtts_vocab_path = f"{PROJECT_ROOT}/{self.config['tts_vocab_path']}"

        self.logger.info("Loading XTTS model...")
        config = XttsConfig()
        config.load_json(xtts_config_path)

        self.tts = Xtts.init_from_config(config)
        self.tts.load_checkpoint(
            config=config,
            checkpoint_path=xtts_model_path,
            vocab_path=xtts_vocab_path,
            speaker_file_path=f"{PROJECT_ROOT}/{self.config['tts_speakers']}",
            use_deepspeed=False,
        )

        if self.device == "cuda":
            self.tts.cuda()

        self.logger.info("XTTS model loaded successfully.")

    def infer_audio(self, text: str) -> str:

        self.logger.info("Starting TTS inference...")
        speaker_audio_path = f"{PROJECT_ROOT}/{self.config['tts_reference_wav']}"
        gpt_cond_latent, speaker_embedding = self.tts.get_conditioning_latents(
            audio_path=speaker_audio_path,
            gpt_cond_len=self.tts.config.gpt_cond_len,
            max_ref_length=self.tts.config.max_ref_len,
            sound_norm_refs=self.tts.config.sound_norm_refs,
        )

        out = self.tts.inference(
            text=text,
            language=self.config["tts_language"],
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=self.tts.config.temperature,
            length_penalty=self.tts.config.length_penalty,
            repetition_penalty=self.tts.config.repetition_penalty,
            top_k=self.tts.config.top_k,
            top_p=self.tts.config.top_p,
            enable_text_splitting=True,
        )

        output_file_name = f"{''.join(random.choices(string.ascii_letters, k=7))}.wav"
        output_file_path = f"{UNCONVERTED_DATA_DIR}/{output_file_name}"

        out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
        torchaudio.save(output_file_path, out["wav"], 24000)

        self.logger.info(f"TTS synthesis complete. Audio saved at: {output_file_path}")
        return output_file_name

