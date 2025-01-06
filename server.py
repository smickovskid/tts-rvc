from flask import Flask, abort, jsonify, request, send_file
from dotenv import load_dotenv
import logging


from rvc_wrapper.client import RVCWrapper
from tts_wrapper.xtts_client import XTTSWrapper
from utils.logger import setup_logger


class RVCFlaskApp:
    def __init__(self):
        self.LOGGER = setup_logger(log_level=logging.INFO)
        load_dotenv()
        self.app = Flask(__name__)

        # TTS setup
        self.tts_wrapper = XTTSWrapper(self.LOGGER)
        self.tts_config = self.tts_wrapper.config

        # Rvc setup
        self.rvc_wrapper = RVCWrapper(self.LOGGER)
        self.rvc_config = self.rvc_wrapper.config

        self._setup_routes()

    def _setup_routes(self):
        """Define all Flask routes."""

        @self.app.route("/health", methods=["GET"])
        def health():
            if self.tts_wrapper is not None:
                return jsonify({"status": "ready"})
            abort(503)

        @self.app.route("/generate", methods=["POST"])
        def generate():
            try:
                request_data = request.get_json()

                if not request_data or "message" not in request_data:
                    abort(400, description="Missing 'message' in the request JSON")

                self.LOGGER.info(request_data["message"])
                audio = self.tts_wrapper.infer_audio(request_data["message"])
                ref_audio = self.rvc_wrapper.infer_audio(audio)

                return send_file(
                    ref_audio,
                    as_attachment=True,
                    download_name="out_from_text.wav",  # Specify the download filename
                    mimetype="audio/wav",
                )
            except Exception as e:
                self.LOGGER.error(e)
                abort(503)

    def run(self):
        """Run the Flask app."""
        self.app.run(
            debug=False, host=self.rvc_config["HOST"], port=int(self.rvc_config["PORT"])
        )


if __name__ == "__main__":
    app_instance = RVCFlaskApp()
    app_instance.run()
