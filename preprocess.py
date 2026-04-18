import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical # type: ignore

from data_collection import actions, no_sequences, sequence_length, DATA_PATH

# turns text labels into numbers
label_map = {label:num for num, label in enumerate(actions)}

def load_data():
    sequences, labels = [], []
    
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                # Load each individual frame
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            
            # Store the 30-frame video in sequences
            sequences.append(window)
            # Store the corresponding action number in labels
            labels.append(label_map[action])
            
    # Convert to numpy arrays
    X = np.array(sequences)
    
    # One-hot encoding
    Y = to_categorical(labels).astype(int)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20,random_state=42, stratify=labels)
    
    return X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test = load_data()

if __name__ == "__main__":
    print("Preprocess.py executed directly.")
    print(f"Total actions: {actions}")
    print(f"X_train shape: {X_train.shape}") 
    print(f"Y_train shape: {Y_train.shape}") 