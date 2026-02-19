import numpy as np
from src.data_loader import prepare_data
from src.scalogram_generator import generate_scalogram
from src.model import build_cnn_model
import os
from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint("models/ecg_cnn_model.h5", save_best_only=False)
callbacks=[checkpoint]

BATCH_SIZE = 16
EPOCHS = 30

X_train, X_test, y_train, y_test = prepare_data()

def generator(X, y, batch=16):
    while True:
        for i in range(0, len(X), batch):
            Xb, yb = [], y[i:i+batch]
            for beat in X[i:i+batch]:
                scalo = generate_scalogram(beat)
                Xb.append(scalo)
            Xb = np.array(Xb)[..., np.newaxis]
            yield Xb, yb

model = build_cnn_model()

model.fit(
    generator(X_train, y_train, BATCH_SIZE),
    validation_data=generator(X_test, y_test, BATCH_SIZE),
    steps_per_epoch=len(X_train)//BATCH_SIZE,
    validation_steps=len(X_test)//BATCH_SIZE,
    epochs=EPOCHS
)

os.makedirs("models", exist_ok=True)
model.save("models/ecg_cnn_model.h5")
