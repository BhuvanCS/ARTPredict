from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ParameterGrid
import ast 

def load_encoded_sequences(sequence_column):
    # Convert string representation of array back to numpy array
    return np.array([ast.literal_eval(seq) for seq in sequence_column])

def create_model(input_shape_num, input_shape_seq, l2_reg):
    # Input for numerical data
    numerical_input = Input(shape=(input_shape_num,), name='numerical_input')
    x_num = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(numerical_input)
    x_num = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x_num)

    # Input for one-hot encoded sequence data
    sequence_input = Input(shape=(input_shape_seq, 4), name='sequence_input')  # Shape: (sequence_length, 4)
    x_seq = LSTM(32)(sequence_input)

    # Combine the numerical and sequence parts
    combined = concatenate([x_num, x_seq])
    x = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(combined)
    x = Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    output = Dense(1, activation='sigmoid')(x)  # Assuming binary classification (0/1)

    # Create the model
    model = Model(inputs=[numerical_input, sequence_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def hyperparameter_tuning(preprocessed_data_path, hyperparams=None):
    # Load the preprocessed data
    data = pd.read_csv(preprocessed_data_path)

    # Separate features and target
    X_numerical = data[['Viral_Load', 'CD4_Count', 'Adherence_Level', 'HIV_Strain']]
    y = data['Treatment_Response']

    # Load and decode the sequence data
    sequences = load_encoded_sequences(data['Encoded_Sequence'].astype(str).values)

    # Check that the sequences are in the expected format
    if sequences.ndim != 3 or sequences.shape[2] != 4:
        raise ValueError("Sequence data should be in one-hot encoded format with shape (samples, time_steps, 4).")

    # Determine the maximum sequence length
    max_seq_length = sequences.shape[1]

    # Split the data into train and test sets
    X_train_num, X_test_num, X_train_seq, X_test_seq, y_train, y_test = train_test_split(
        X_numerical, sequences, y, test_size=0.2, random_state=42
    )

    # Hyperparameter tuning using ParameterGrid
    best_accuracy = 0
    best_params = None

    for params in ParameterGrid(hyperparams):
        print(f"Training with params: {params}")
        l2_reg = params.get('l2_reg', 0.01)  # Default value for L2 regularization
        batch_size = params.get('batch_size', 32)

        # Create model with current hyperparameters
        model = create_model(X_train_num.shape[1], max_seq_length, l2_reg)

        # Train the model
        model.fit(
            [X_train_num, X_train_seq], y_train,
            epochs=30, batch_size=batch_size, verbose=0
        )

        # Evaluate the model
        _, accuracy = model.evaluate([X_test_num, X_test_seq], y_test, verbose=0)
        print(f"Neural Network model accuracy: {accuracy}")

        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params

    print(f"Best accuracy: {best_accuracy} with params: {best_params}")

# Example usage
preprocessed_file_path = 'app/data/preprocessed_data/preprocessed_clinical_data.csv'

# Define the hyperparameter grid
hyperparameter_grid = {
    'l2_reg': [0.01, 0.001, 0.0001],  # L2 regularization values
    'batch_size': [16, 32, 64],  # Batch sizes to try
}

hyperparameter_tuning(preprocessed_file_path, hyperparams=hyperparameter_grid)
