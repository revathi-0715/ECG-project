import os
import wfdb
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from src.scalogram_generator import generate_scalogram

CLASSES = ["Normal", "LBBB", "RBBB", "PVC", "APC"]


def load_record_for_prediction(record_path):
    """Load ECG record and extract individual beats."""
    hea = record_path + ".hea"
    dat = record_path + ".dat"

    if not os.path.exists(hea) or not os.path.exists(dat):
        raise FileNotFoundError(
            f"Required files missing for record: {record_path}"
        )

    record = wfdb.rdrecord(record_path, physical=False)
    signal = record.d_signal[:, 0]

    try:
        ann = wfdb.rdann(record_path, "atr")
    except Exception:
        ann = wfdb.rdann(record_path, "qrs")

    beats = []
    for p in ann.sample:
        start, end = max(0, p - 200), min(len(signal), p + 200)
        beat = signal[start:end]
        if len(beat) < 400:
            beat = np.pad(beat, (0, 400 - len(beat)))
        beats.append(beat)

    return beats


def save_scalogram_image(scalogram, save_path, predicted_class):
    """Save the scalogram of the representative beat as a PNG image."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(scalogram, aspect="auto", cmap="jet", origin="lower")
    ax.set_title(f"ECG Scalogram — Predicted: {predicted_class}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Scale / Frequency")
    plt.colorbar(ax.images[0], ax=ax, label="Magnitude")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)


def predict_record(model_path, record_path, scalogram_save_path=None):
    """
    Predict arrhythmia class for an ECG record.

    Args:
        model_path: Path to the trained .h5 model.
        record_path: Path to the WFDB record (without extension).
        scalogram_save_path: Optional path to save the representative scalogram image.

    Returns:
        Predicted class label string.
    """
    model = load_model(model_path)
    beats = load_record_for_prediction(record_path)

    predictions = []
    scalograms = []

    for beat in beats:
        scalogram = generate_scalogram(beat)
        scalograms.append(scalogram)
        inp = scalogram[np.newaxis, ..., np.newaxis]
        pred = model.predict(inp, verbose=0)
        predictions.append(np.argmax(pred))

    # Majority voting across all beats
    final_class_idx = max(set(predictions), key=predictions.count)
    predicted_class = CLASSES[final_class_idx]

    # Save the scalogram of the first beat predicted as the majority class
    if scalogram_save_path:
        # Pick first beat that matches the majority prediction
        rep_idx = next(
            (i for i, p in enumerate(predictions) if p == final_class_idx), 0
        )
        save_scalogram_image(scalograms[rep_idx], scalogram_save_path, predicted_class)

    return predicted_class