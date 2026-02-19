from src.data_loader import prepare_data
from src.scalogram_generator import generate_scalograms_for_dataset
from src.model import train_model
from src.utils import evaluate_model
from src.predict import predict_beat
import numpy as np

records = ['100', '101', '102', '103', '104']  # 5 different records from your data
X_train, X_test, y_train_cat, y_test_cat = prepare_data(records)
X_train_scalo = generate_scalograms_for_dataset(X_train)
X_test_scalo = generate_scalograms_for_dataset(X_test)
model = train_model(X_train_scalo, y_train_cat, X_test_scalo, y_test_cat)
evaluate_model(model, X_test_scalo, y_test_cat, ['Normal', 'LBBB', 'RBBB', 'PVC', 'APC'])
model.save('models/ecg_cnn_model.h5')

# Test on new data (replace with real beat from unseen record, e.g., '105')
new_beat = np.random.randn(400)
pred, probs = predict_beat('models/ecg_cnn_model.h5', new_beat)
print(f"New Data Prediction: {pred}")