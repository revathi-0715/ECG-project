import os
import json
import wfdb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Force TensorFlow to use tf-keras (Keras 2) backend
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
from src.scalogram_generator import generate_scalogram

CLASSES = ["Normal", "LBBB", "RBBB", "PVC", "APC"]


def load_model_safe(model_path):
    """Load Keras 3 saved model using multiple fallback strategies."""

    # ── Strategy 1: use tf_keras directly ───────────────────────────────────
    try:
        import tf_keras
        model = tf_keras.models.load_model(model_path, compile=False)
        print("Model loaded via tf_keras.")
        return model
    except Exception as e1:
        print(f"Strategy 1 failed: {e1}")

    # ── Strategy 2: patch DTypePolicy then load ──────────────────────────────
    try:
        from tensorflow.python.keras.mixed_precision.policy import Policy

        class DTypePolicy(Policy):
            def __init__(self, name, **kwargs):
                super().__init__(name)

        custom_objects = {
            "DTypePolicy": DTypePolicy,
            "Orthogonal": tf.keras.initializers.Orthogonal,
        }

        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(model_path, compile=False)
        print("Model loaded via Strategy 2.")
        return model
    except Exception as e2:
        print(f"Strategy 2 failed: {e2}")

    # ── Strategy 3: fix h5 config JSON directly ──────────────────────────────
    try:
        import h5py

        with h5py.File(model_path, 'r') as f:
            model_config = f.attrs.get('model_config')
            if isinstance(model_config, bytes):
                model_config = model_config.decode('utf-8')
            config = json.loads(model_config)

        def fix_config(obj):
            """Recursively fix Keras 3 specific config keys."""
            if isinstance(obj, dict):
                # Fix DTypePolicy → plain string dtype
                if obj.get('class_name') == 'DTypePolicy':
                    return obj.get('config', {}).get('name', 'float32')
                # Fix dtype dict to plain string
                if 'dtype' in obj and isinstance(obj['dtype'], dict):
                    if obj['dtype'].get('class_name') == 'DTypePolicy':
                        obj['dtype'] = obj['dtype'].get('config', {}).get('name', 'float32')
                # Fix batch_shape → input_shape
                if 'batch_shape' in obj:
                    obj['input_shape'] = obj.pop('batch_shape')[1:]
                # Fix module-based class references
                for key in ['kernel_initializer', 'bias_initializer',
                            'activation', 'kernel_regularizer']:
                    if key in obj and isinstance(obj[key], dict):
                        if 'module' in obj[key]:
                            obj[key].pop('module', None)
                            obj[key].pop('registered_name', None)
                return {k: fix_config(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [fix_config(i) for i in obj]
            return obj

        fixed_config = fix_config(config)
        model = tf.keras.models.model_from_json(json.dumps(fixed_config))

        with h5py.File(model_path, 'r') as f:
            model.load_weights(model_path)

        print("Model loaded via Strategy 3 (config repair).")
        return model
    except Exception as e3:
        raise RuntimeError(
            f"All strategies failed.\n"
            f"S1: {e1}\nS2: {e2}\nS3: {e3}"
        )


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
    """Save the scalogram as a PNG image."""
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(scalogram, aspect="auto", cmap="jet", origin="lower")
    ax.set_title(
        f"ECG Scalogram — Predicted: {predicted_class}",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Scale / Frequency")
    plt.colorbar(im, ax=ax, label="Magnitude")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)


def predict_record(model_path, record_path, scalogram_save_path=None):
    """Predict arrhythmia class for an ECG record."""
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
