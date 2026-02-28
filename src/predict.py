import os
import wfdb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from src.scalogram_generator import generate_scalogram

CLASSES = ["Normal", "LBBB", "RBBB", "PVC", "APC"]


def load_model_safe(model_path):
    """Load model with fallback for version compatibility issues."""
    try:
        # First try: standard load
        model = load_model(model_path)
        return model
    except Exception:
        pass

    try:
        # Second try: compile=False skips optimizer config issues
        model = load_model(model_path, compile=False)
        return model
    except Exception:
        pass

    try:
        # Third try: custom_objects to fix batch_shape issue
        from tensorflow.keras.layers import InputLayer

        class CompatInputLayer(InputLayer):
            def __init__(self, **kwargs):
                if 'batch_shape' in kwargs:
                    kwargs['input_shape'] = kwargs.pop('batch_shape')[1:]
                super().__init__(**kwargs)

        model = load_model(
            model_path,
            compile=False,
            custom_objects={'InputLayer': CompatInputLayer}
        )
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")


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
    im = ax.imshow(scalogram, aspect="auto", cmap="jet", origin="lower")
    ax.set_title(f"ECG Scalogram — Predicted: {predicted_class}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Scale / Frequency")
    plt.colorbar(im, ax=ax, label="Magnitude")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)


def predict_record(model_path, record_path, scalogram_save_path=None):
    """
    Predict arrhythmia class for an ECG record.
    """
    model = load_model_safe(model_path)
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

    # Save scalogram of first beat matching majority class
    if scalogram_save_path:
        rep_idx = next(
            (i for i, p in enumerate(predictions) if p == final_class_idx), 0
        )
        save_scalogram_image(scalograms[rep_idx], scalogram_save_path, predicted_class)

    return predicted_class
