from flask import Flask, abort, request, send_file
from flask_restx import Api, Resource
from dotenv import load_dotenv
import logging

from rvc.modules.vc.modules import os


from rvc_wrapper.client import RVCWrapper
from tts_wrapper.xtts_client import XTTSWrapper
from api_models.models import define_models
from utils.logger import setup_logger


LOGGER = setup_logger(log_level=logging.INFO)
load_dotenv()
app = Flask(__name__)

# TTS setup
tts_wrapper = XTTSWrapper(LOGGER)
tts_config = tts_wrapper.config

# RVC setup
rvc_wrapper = RVCWrapper(LOGGER)
rvc_config = rvc_wrapper.config

api = Api(app)
models = define_models(api)

@api.route("/health")
class Health(Resource):
    @api.doc(responses={200: "Service is ready", 503: "Service is unavailable"})
    @api.marshal_with(models["health_response_model"])
    def get(self):
        if tts_wrapper is not None:
            return {"status": "ready"}, 200
        abort(503)

@api.route("/generate")
@api.doc(params={'message': 'Message to infer'})
class Generate(Resource):
    @api.expect(models["generate_request_model"])
    @api.doc(
        responses={
            200: ("Audio file generated successfully", "audio/wav"),
            400: "Invalid request: Message is missing",
            503: "Service unavailable: Internal server error",
        }
    )
    def post(self):
        try:
            request_data = request.get_json()

            if not request_data or "message" not in request_data:
                return {"error": "Message missing from body."}, 400

            LOGGER.debug(request_data["message"])
            audio = tts_wrapper.infer_audio(request_data["message"])
            ref_audio = rvc_wrapper.infer_audio(audio)

            return send_file(
                ref_audio,
                as_attachment=True,
                download_name="out_from_text.wav",
                mimetype="audio/wav",
            )
        except Exception as e:
            LOGGER.error(e)
            return {"error": "Something went wrong."}, 503

if __name__ == "__main__":
        HOST = os.getenv("HOST", "127.0.0.1")
        PORT = os.getenv("PORT", "5500")

        app.run(
            debug=True, host=HOST, port=int(PORT)
        )

        LOGGER.info(f"Server running on {HOST}:{PORT}")
