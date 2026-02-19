import os
import uuid
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from src.predict import predict_record

DATASET_DIR = "data/mitdb/x_mitdb"
MODEL_PATH = "models/ecg_cnn_model.h5"
SCALOGRAM_SAVE_DIR = "static/scalograms"

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

        # Unique filename to avoid caching conflicts
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
    app.run(debug=True)