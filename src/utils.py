import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import numpy as np

def plot_scalogram(scalogram):
    plt.imshow(scalogram.squeeze(), cmap='viridis')
    plt.title("Scalogram")
    plt.show()

def evaluate_model(model, X_test, y_test, class_names):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    print(classification_report(y_true, y_pred, target_names=class_names))
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"Weighted F1-Score: {f1:.4f}")
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, xticklabels=class_names, yticklabels=class_names)
    plt.show()