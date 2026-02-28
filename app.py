import os
import uuid
import gdown
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from src.predict import predict_record

DATASET_DIR = "data/mitdb/x_mitdb"
MODEL_PATH = "heartbeat_classifier.h5"
SCALOGRAM_SAVE_DIR = "static/scalograms"

# ── Auto-download model from Google Drive if not present ──────────────────────
GDRIVE_FILE_ID = "1rjl9BSI8Pb3q8pUr_pUYYlmjCkYJ3CUP"  # <-- Replace this with your file ID

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Model not found locally. Downloading from Google Drive...")
        os.makedirs("models", exist_ok=True)
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
        print("Model downloaded successfully.")
    else:
        print("Model already exists. Skipping download.")

download_model()
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
os.makedirs(SCALOGRAM_SAVE_DIR, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    scalogram_image = None

    if request.method == "POST":
        file = request.files["file"]
        filename = secure_filename(file.filename)
        record_name = filename.split(".")[0]
        record_path = os.path.join(DATASET_DIR, record_name)

        unique_id = uuid.uuid4().hex[:8]
        scalogram_filename = f"scalogram_{record_name}_{unique_id}.png"
        scalogram_save_path = os.path.join(SCALOGRAM_SAVE_DIR, scalogram_filename)

        try:
            prediction = predict_record(MODEL_PATH, record_path, scalogram_save_path)
            scalogram_image = f"scalograms/{scalogram_filename}"
        except FileNotFoundError as e:
            error = str(e)
        except Exception as e:
            error = f"Prediction failed: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        scalogram_image=scalogram_image,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
