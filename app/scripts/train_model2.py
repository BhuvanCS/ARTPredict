from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, concatenate, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import ast 

def load_encoded_sequences(sequence_column):
    # Convert string representation of array back to numpy array
    return np.array([ast.literal_eval(seq) for seq in sequence_column])

def train_and_save_nn_model(preprocessed_data_path, model_output_path='app/models/neural_network_model.h5', 
                             x_train_path='app/models/x_train_nn.npy', x_test_path='app/models/x_test_nn.npy', 
                             y_test_path='app/models/y_test_nn.npy'):
    # Load the preprocessed data
    data = pd.read_csv(preprocessed_data_path)

    # Separate features and target
    X_numerical = data[['Viral_Load', 'CD4_Count', 'Adherence_Level', 'HIV_Strain']]
    y = data['Treatment_Response']

    # Load and decode the sequence data
    sequences = load_encoded_sequences(data['Encoded_Sequence'].astype(str).values)
    sequence_data = np.array(data['Encoded_Sequence'].tolist())
    sequence_data = sequences

    # Check that the sequences are in the expected format
    if sequences.ndim != 3 or sequences.shape[2] != 4:
        raise ValueError("Sequence data should be in one-hot encoded format with shape (samples, time_steps, 4).")

    # Determine the maximum sequence length
    max_seq_length = sequences.shape[1]

    # Split the data into train and test sets
    X_train_num, X_test_num, X_train_seq, X_test_seq, y_train, y_test = train_test_split(
        X_numerical, sequence_data, y, test_size=0.2, random_state=42
    )

    # Build the neural network model
    # Input for numerical data
    numerical_input = Input(shape=(X_train_num.shape[1],), name='numerical_input')
    x_num = Dense(64, activation='relu')(numerical_input)
    x_num = Dense(32, activation='relu')(x_num)
    vocab_size = 10
    # Input for one-hot encoded sequence data
    sequence_input = Input(shape=(max_seq_length, 4), name='sequence_input')  # Shape: (sequence_length, 4)
    seq_lstm = LSTM(64, return_sequences=False)(sequence_input)
    x_seq = LSTM(32)(sequence_input)

    # Combine the numerical and sequence parts
    combined = concatenate([x_num, x_seq])
    x = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001))(combined)
    x = Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    output = Dense(1, activation='sigmoid')(x)  # Assuming binary classification (0/1)

    # Create the model
    model = Model(inputs=[numerical_input, sequence_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    # Set up early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    model.fit(
        [X_train_num, X_train_seq], y_train,
        validation_data=([X_test_num, X_test_seq], y_test),
        epochs=30, batch_size=16
    )

    predictions = model.predict([X_test_num, X_test_seq]).flatten()
    print(predictions)
    responder_cases = X_test_num[predictions > 0.5]  # Adjust threshold as needed for classification
    non_responder_cases = X_test_num[predictions < 0.5]  # Adjust threshold as needed for classification
    print(responder_cases)
    print("Non-resp")
    print(non_responder_cases)

    print("Test cases predicted as 'Responder':")
    for i, case in enumerate(responder_cases):
        print(f"Case {i + 1}: Numerical Data = {case}, Predicted Probability = {predictions[i]:.2f}")


    # Evaluate and save the model
    _, accuracy = model.evaluate([X_test_num, X_test_seq], y_test, verbose=0)
    print(f"Neural Network model accuracy: {accuracy}")

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    model.save(model_output_path)
    print(f"Neural Network model saved to {model_output_path}")

    # Save tokenizer and other data for future use
    joblib.dump(None, os.path.join('app/models', 'tokenizer.pkl'))  # Placeholder since you don't need tokenizer now
    np.save(x_train_path, X_train_num)
    np.save(x_test_path, X_test_num)
    np.save(x_train_path.replace('x_train_nn', 'x_train_seq_nn'), X_train_seq)
    np.save(x_test_path.replace('x_test_nn', 'x_test_seq_nn'), X_test_seq)
    np.save(y_test_path, y_test)

# Example usage
preprocessed_file_path = 'app/data/preprocessed_data/preprocessed_clinical_data.csv'
train_and_save_nn_model(preprocessed_file_path)
