import os
import string
import random
from typing import TypedDict
from logging import Logger
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Path, Xtts, torchaudio
from utils.config import check_env_vars
from utils.const import PROJECT_ROOT, UNCONVERTED_DATA_DIR


class XTTSV2Config(TypedDict):
    tts_model_dir: str
    tts_language: str
    use_deepspeed: bool


class XTTSWrapper:
    def __init__(self, logger: Logger):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logger
        self.config: XTTSV2Config = {
            "tts_model_dir": os.getenv("tts_model_dir", ""),
            "tts_language": os.getenv("tts_language", ""),
            "use_deepspeed": bool(os.getenv("use_deepspeed", "").lower() in ["true", "1", "yes"]),
        }
        check_env_vars(dict(self.config), self.logger)

        # Load XTTS configuration and model
        xtts_config_path = f"{PROJECT_ROOT}/{self.config['tts_model_dir']}/config.json"
        xtts_model_path = f"{PROJECT_ROOT}/{self.config['tts_model_dir']}/model.pth"
        xtts_vocab_path = f"{PROJECT_ROOT}/{self.config['tts_model_dir']}/vocab.json"
        xtts_speakers_path = f"{PROJECT_ROOT}/{self.config['tts_model_dir']}/speakers_xtts.pth"
        reference_dir = Path(f"{PROJECT_ROOT}/{self.config['tts_model_dir']}/reference_files")

        for path in [xtts_config_path, xtts_model_path, xtts_vocab_path]:
            if not Path(path).exists():
                raise FileNotFoundError(f"Required file does not exist: {path}")

        if xtts_speakers_path and not Path(xtts_speakers_path).exists():
            self.logger.warning(f"Speakers path does not exist, proceeding without it")

        self.reference_files = [str(file) for file in reference_dir.glob("*.wav")]
        if not self.reference_files:
            raise FileNotFoundError(f"No .wav files found in reference directory: {reference_dir}")

        self.logger.info("Loading XTTS model...")
        config = XttsConfig()
        config.load_json(xtts_config_path)

        self.tts = Xtts.init_from_config(config)
        self.tts.load_checkpoint(
            config=config,
            checkpoint_path=xtts_model_path,
            vocab_path=xtts_vocab_path,
            speaker_file_path=xtts_speakers_path,
            use_deepspeed=self.config["use_deepspeed"],
        )

        if self.device == "cuda":
            self.tts.cuda()

        self.logger.info("XTTS model loaded successfully.")
        self.logger.info("Warming up model")
        _ = self.infer_audio("Generating warmup text so the first response is fast.")
        self.logger.info("Warmup done")

    def infer_audio(self, text: str) -> str:

        self.logger.info("Starting TTS inference...")
        gpt_cond_latent, speaker_embedding = self.tts.get_conditioning_latents(
            audio_path=self.reference_files,
            gpt_cond_len=self.tts.config.gpt_cond_len,
            max_ref_length=self.tts.config.max_ref_len,
            sound_norm_refs=self.tts.config.sound_norm_refs,
        )
        temperature = self.tts.config.temperature
        length_penalty = self.tts.config.length_penalty
        repetition_penalty = self.tts.config.repetition_penalty

        out = self.tts.inference(
            text=text,
            language=self.config["tts_language"],
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
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

