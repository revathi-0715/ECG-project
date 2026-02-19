import wfdb
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import random
import os

# ==============================
# Configuration
# ==============================

DATA_DIR = r"data/mitdb/x_mitdb"

CLASSES = {'N': 0, 'L': 1, 'R': 2, 'V': 3, 'A': 4}
CLASS_NAMES = ['Normal', 'LBBB', 'RBBB', 'PVC', 'APC']

WINDOW_SIZE = 400

# ==============================
# Data Augmentation
# ==============================

def augment_beat(beat, noise_factor=0.01, shift_max=10):
    noise = np.random.normal(0, noise_factor * np.std(beat), beat.shape)
    beat = beat + noise
    shift = random.randint(-shift_max, shift_max)
    return np.roll(beat, shift)

# ==============================
# Load Single MIT-BIH Record
# ==============================

def load_mitbih_record(record_name, augment=False):
    path = os.path.join(DATA_DIR, record_name)

    record = wfdb.rdrecord(path, physical=False)
    signal = record.d_signal[:, 0].astype(np.float32)
    ann = wfdb.rdann(path, 'atr')

    beats, labels = [], []

    for i, symbol in enumerate(ann.symbol):
        if symbol in CLASSES:
            p = ann.sample[i]
            start = max(0, p - WINDOW_SIZE // 2)
            end = min(len(signal), p + WINDOW_SIZE // 2)

            beat = signal[start:end]

            if len(beat) < WINDOW_SIZE:
                beat = np.pad(beat, (0, WINDOW_SIZE - len(beat)))

            beats.append(beat)
            labels.append(CLASSES[symbol])

            # Augment minority classes
            if augment and symbol in ['L', 'R', 'V', 'A']:
                beats.append(augment_beat(beat))
                labels.append(CLASSES[symbol])

    return beats, labels

# ==============================
# Prepare Dataset
# ==============================

def prepare_data(test_size=0.2):
    records = sorted(set(f.split('.')[0] for f in os.listdir(DATA_DIR) if f.endswith('.dat')))

    X, y = [], []

    print(f"Loading {len(records)} records...")

    for r in records:
        beats, labels = load_mitbih_record(r, augment=True)
        X.extend(beats)
        y.extend(labels)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    # Normalize
    X = (X - np.mean(X)) / np.std(X)

    # One-hot encoding
    y = to_categorical(y, num_classes=5)

    # Safe stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        shuffle=True,
        stratify=np.argmax(y, axis=1)
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    return X_train, X_test, y_train, y_test

