import os
from tensorflow.keras.models import Sequential           # type: ignore 
from tensorflow.keras.layers import LSTM, Dense         # type: ignore
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping # type: ignore
from tensorflow.keras.optimizers import Adam             # type: ignore

from data_collection import actions, no_sequences, sequence_length, DATA_PATH
from preprocess import X_train, X_test, Y_train, Y_test

def build_model():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(30, 150)))
    model.add(LSTM(128, return_sequences=True, activation='tanh'))
    model.add(LSTM(64, return_sequences=False, activation='tanh'))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(actions.shape[0], activation="softmax"))
    return model

if __name__ == "__main__":
    log_dir = os.path.join("Logs")
    tb_callback = TensorBoard(log_dir=log_dir)
    early_stopping = EarlyStopping(monitor='categorical_accuracy', patience=50, restore_best_weights=True)

    model = build_model()
    model.compile(
        optimizer=Adam(learning_rate=0.0001), 
        loss='categorical_crossentropy', 
        metrics=['categorical_accuracy']
    )

    print("Starting training... Press Ctrl+C to stop.")
    model.fit(X_train, Y_train, epochs=1000, callbacks=[tb_callback, early_stopping])

    model.save('action.h5')