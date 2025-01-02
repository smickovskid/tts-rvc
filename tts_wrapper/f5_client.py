import os
import string
import random
from typing import TypedDict
from logging import Logger
import torch
import torchaudio
from cached_path import cached_path
from importlib.resources import files

from f5_tts.infer.utils_infer import infer_batch_process, preprocess_ref_audio_text, load_vocoder, load_model
from f5_tts.model.backbones.dit import DiT
from utils.config import check_env_vars
from utils.const import PROJECT_ROOT, UNCONVERTED_DATA_DIR


class F5TTSConfig(TypedDict):
    ckpt_file: str
    vocab_file: str
    ref_audio: str
    ref_text: str


class F5TTSWrapper:
    def __init__(self, logger: Logger):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logger
        self.config: F5TTSConfig = {
            "ckpt_file": os.getenv("f5_ckpt_file", ""),
            "vocab_file": os.getenv("f5_vocab_file", ""),
            "ref_audio": os.getenv("tts_reference_wav", ""),
            "ref_text": os.getenv("tts_reference_text", ""),
        }
        check_env_vars(self.config, self.logger)

        # Load model paths
        ckpt_file_path = self.config["ckpt_file"]
        vocab_file_path = self.config["vocab_file"]
        ref_audio_path = self.config["ref_audio"]
        ref_text = self.config["ref_text"]

        # Load the model
        self.logger.info("Loading TTS model...")
        self.model = load_model(
            model_cls=DiT,
            model_cfg=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
            ckpt_path=ckpt_file_path,
            mel_spec_type="vocos",  # or "bigvgan" depending on vocoder
            vocab_file=vocab_file_path,
            ode_method="euler",
            use_ema=True,
            device=self.device,
        ).to(self.device, dtype=torch.float32)

        # Load the vocoder
        self.vocoder = load_vocoder(is_local=False)

        # Store reference paths
        self.ref_audio = ref_audio_path
        self.ref_text = ref_text

        # Warm up the model
        self._warm_up()
        self.logger.info("TTS model loaded successfully.")

    def _warm_up(self):
        """Warm up the model with a dummy input."""
        self.logger.info("Warming up the model...")
        ref_audio, ref_text = preprocess_ref_audio_text(self.ref_audio, self.ref_text)
        audio, sr = torchaudio.load(ref_audio)
        gen_text = "Warm-up text for the model."

        infer_batch_process((audio, sr), ref_text, [gen_text], self.model, self.vocoder, device=self.device)
        self.logger.info("Model warm-up completed.")

    def infer_audio(self, text: str) -> str:
        """Generate audio from the input text."""
        self.logger.info("Starting TTS inference...")
        ref_audio, ref_text = preprocess_ref_audio_text(self.ref_audio, self.ref_text)
        audio, sr = torchaudio.load(ref_audio)

        # Run inference
        audio_chunk, sample_rate, _ = infer_batch_process(
            (audio, sr),
            ref_text,
            [text],
            self.model,
            self.vocoder,
            device=self.device,
            speed=0.8,
            nfe_step=16
        )

        # Generate a random file name and save the audio
        output_file_name = f"{''.join(random.choices(string.ascii_letters, k=7))}.wav"
        output_file_path = f"{UNCONVERTED_DATA_DIR}/{output_file_name}"

        # Save the generated audio
        torchaudio.save(output_file_path, torch.tensor(audio_chunk).unsqueeze(0), sample_rate)

        self.logger.info(f"TTS synthesis complete. Audio saved at: {output_file_path}")
        return output_file_name

