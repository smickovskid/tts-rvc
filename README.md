### Coqui XTTSv2 + RVC pipeline

This pipeline combines the Coqui and RVC SDKs to generate audio. I've created this for myself as an alternative to piper for home assistant voice assistants. Piper was a bit robotic for me so I oped for XTTS with RVC to achieve my desirable results.

### Prerequisites
- `python==3.10`

- XTTSv2 files
    - model.pth
    - vocab.txt
    - config.json
    - speakers_xtts.pth (optional)

- RVC files
    - model.pth
    - model.index

- Reference audio file(s)


### Installation

Either with `venv` or `conda`

#### Semi automated setup
1. `conda create -n xtts-rvc python==3.10`
2. `conda activate xtts-rvc`
3. `./setup.sh --coqui_model_zip <url_to_zip> --rvc_model_zip <url_to_zip> --name <project_name>`
    - The coqui `.zip` needs to contain `config.json`, `vocab.json`, `model.pth`, `speakers_xtts.pth`(optional).
4. `.env` file with the paths to all the files (see `example.env`).
5. Inside `tts_models/<your_model>` create a folder named `reference_files` and put one or multiple `.wav` reference files there for the XTTS inference.
6. `python server.py`

#### Manual setup
1. `conda create -n xtts-rvc python==3.10`
2. `conda activate xtts-rvc`
3. `pip install --upgrade pip==24 # because of pip>24 automatically rejecting dependency conflicts`
4. `pip install -r requirements.txt`
5. For rvc either run [download_models.py](https://raw.githubusercontent.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/refs/heads/main/tools/download_models.py) or follow the readme in their repo on where to get the model files. They all go under the `assets` directory which you need to create in the root of the repo.
6. Create a directory `tts_models` and another directory with your model name inside it.
7. Unzip or place your XTTS model files here.
8. Inside `tts_models/<your_model>` create a folder named `reference_files` and put one or multiple `.wav` reference files there for the XTTS inference.
9. `.env` file with the paths to all the files (see `example.env`).
10. `python server.py`

### Inference
**Request**
`POST /generate`
```json
{
    "message": "Your message"
}
```
**Response**
`.wav` file as bytes.

### Misc
Under the `data` directory you can find 2 sub directories, `converted` and `unconverted` which contain the audio files. The `unconverted` directory contains the generated XTTS audio files whereas `converted` contains the RVC ones.

### Swagger
You can access swagger under the `/` route.

### Reference Projects

Everything related to the models and training is owned by their respective parties.

- [Coqui TTS](https://github.com/coqui-ai/TTS)
- [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
