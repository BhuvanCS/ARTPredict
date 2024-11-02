from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_and_save_nn_model(preprocessed_data_path, model_output_path='app/models/neural_network_model.h5', x_train_path='app/models/x_train_nn.npy', x_test_path = 'app/models/x_test_nn.npy', y_test_path = 'app/models/y_test_nn.npy'):
    data = pd.read_csv(preprocessed_data_path)

    X = data.drop(columns=['Treatment_Response', 'Patient_ID'], axis=1)
    y = data['Treatment_Response']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    global global_x_train
    global_x_train = X_train

    # Build a neural network model
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification (0/1)

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    # Set up early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, callbacks=[early_stopping])

    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Neural Network model accuracy: {accuracy}")

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    model.save(model_output_path)
    print(f"Neural Network model saved to {model_output_path}")

    np.save(x_train_path, X_train)
    np.save(x_test_path, X_test)
    np.save(y_test_path, y_test)

preprocessed_file_path = 'app/data/preprocessed_data/preprocessed_clinical_data.csv'
train_and_save_nn_model(preprocessed_file_path)