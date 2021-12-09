import os

PATH_TO_DATA = os.path.dirname(__file__)

# VOSK
VOSK_WEIGHTS = ["https://alphacephei.com/kaldi/models/vosk-model-ru-0.10.zip",
                "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.15.zip"]
VOSK_DIR = os.path.join(PATH_TO_DATA, "weights")


def download_vosk(small=True, verbose=False):
    import urllib.request
    import zipfile

    if verbose:
        print("Creating folder [weights]...")
    os.makedirs(VOSK_DIR, exist_ok=True)
    asr_model_path = VOSK_WEIGHTS[1] if small else VOSK_WEIGHTS[0]

    if verbose:
        print("Downloading Vosk...")
    response = urllib.request.urlopen(asr_model_path)
    binary_file = response.read()

    with open(f'{PATH_TO_DATA}/weights/model.zip', 'wb') as writer:
        writer.write(binary_file)

    with zipfile.ZipFile(os.path.join(VOSK_DIR, "model.zip"), "r") as zip_ref:
        zip_ref.extractall(VOSK_DIR)
    os.rename(os.path.join(VOSK_DIR, "vosk-model-small-ru-0.15"),
              os.path.join(VOSK_DIR, "vosk-model-small-ru"))


def check_and_load(verbose=True):
    if not os.path.exists("./weights/vosk-model-small-ru"):
        download_vosk(verbose=verbose)
    if verbose:
        print("All models are downloaded")


if __name__ == "__main__":
    check_and_load()