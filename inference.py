import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

from data_collection import actions
from preprocess import X_test, Y_test

model= load_model('action.h5')

res= model.predict(X_test)
print(res)

# Convert predictions from probabilities to class indices (0, 1, or 2)
yhat = np.argmax(res, axis=1)
# Convert true labels from one-hot encoding to class indices
ytrue = np.argmax(Y_test, axis=1)

# Accuracy
print()
print(f"OVERALL ACCURACY: {accuracy_score(ytrue, yhat) * 100:.2f}%")

# Confusion Matrix
print("\nCONFUSION MATRIX:")
print(multilabel_confusion_matrix(ytrue, yhat))

# Visual comparison of the first few results
print("\nDETAILED COMPARISON (First 5 samples):")
for i in range(min(5, len(yhat))):
    print(f"Sample {i+1} \nPredicted: {actions[yhat[i]]} \nActual: {actions[ytrue[i]]}")