import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Flatten, Concatenate, Attention, Layer, Lambda, MultiHeadAttention
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import os
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

# Load the dataset
data = pd.read_csv("app/data/preprocessed_data/preprocessed_clinical_data_2.csv")

# Define the features and target
numerical_features = ['Viral_Load', 'CD4_Count', 'Adherence_Level', 'Strain_Type']  # Add any other numerical features
sequence_features = [f'Sequence_{i}' for i in range(1, 401)]  # Adjust if the sequences are different in your dataset
target = 'Treatment_Response'

# Preprocess the data
# Assuming 'Treatment_Response' is a categorical variable (e.g., 'Responder', 'Non-Responder')
#data[target] = data[target].map({'Responder': 1, 'Non-Responder': 0})  # Modify as per actual mapping

X_numerical = data[numerical_features].values
X_sequences = data[sequence_features].values


# y = to_categorical(data[target].values, num_classes=2)  # Assuming binary classification (Responder vs Non-Responder)
y = data[target].values
# Normalize the numerical features
scaler = StandardScaler()
X_numerical_scaled = scaler.fit_transform(X_numerical)

# Split data into training and testing sets
X_train_num, X_test_num, X_train_seq, X_test_seq, y_train, y_test = train_test_split(
    X_numerical_scaled, X_sequences, y, test_size=0.2, random_state=42
)

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Build the neural network model
# Numerical input branch
numerical_input = Input(shape=(X_numerical_scaled.shape[1],), name='numerical_input')
x = Dense(64, activation='relu')(numerical_input)
x = Dense(32, activation='relu')(x)
X_sequences = X_sequences.reshape((X_sequences.shape[0], X_sequences.shape[1], 1))
# Sequence input branch
sequence_input = Input(shape=(X_sequences.shape[1], X_sequences.shape[2]), name='sequence_input')
lstm_out = LSTM(64, return_sequences=True)(sequence_input)  # return_sequences=True for Attention layer
attention_output = Attention()([lstm_out, lstm_out])
flattened_attention_output = Flatten()(attention_output)
y = Dense(32, activation='relu')(flattened_attention_output)

print("Shape of attention:", attention_output.shape)
print("Shape of attention map after reduction:", flattened_attention_output.shape)
print("Shape of X_test_seq:", X_test_seq.shape)

# Concatenate the numerical and sequence branches
combined = Concatenate()([x, y])

# Final output layer
output = Dense(2, activation='softmax')(combined)

# Define the model
#model = Model(inputs=[numerical_input, sequence_input], outputs=output)
model = Model(inputs=[numerical_input, sequence_input], outputs=[output, attention_output])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mean_squared_error'])

# Train the model
model.fit([X_train_num, X_train_seq], y_train, epochs=10, batch_size=32, validation_data=([X_test_num, X_test_seq], y_test))

# Evaluate the model
loss, accuracy = model.evaluate([X_test_num, X_test_seq], y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Predict the labels for the test set
y_pred_prob, attention_weights = model.predict([X_test_num, X_test_seq])

# Convert probabilities to class labels (0 or 1)
y_pred = np.argmax(y_pred_prob, axis=1)

# Convert y_test back to class labels (from one-hot encoding to 0 or 1)
y_test_labels = np.argmax(y_test, axis=1)

# Print the confusion matrix
cm = confusion_matrix(y_test_labels, y_pred)
print("Confusion Matrix:")
print(cm)

# Identify and print the samples where the model predicted 0 or 1
pred_zeros = np.where(y_pred == 0)[0]
pred_ones = np.where(y_pred == 1)[0]
print("\nData where model predicted 0 (Non-Responder):")
data_pred_zeros = data.iloc[pred_zeros]
print(data_pred_zeros[numerical_features + sequence_features])  # You can adjust columns to print

# Data associated with predicted 1 (Responder)
print("\nData where model predicted 1 (Responder):")
data_pred_ones = data.iloc[pred_ones]
print(data_pred_ones[numerical_features + sequence_features]) 

model_output_path='app/models/neural_network_model.h5'
x_train_path='app/models/x_train_nn.npy'
x_test_path='app/models/x_test_nn.npy'
y_test_path='app/models/y_test_nn.npy'

os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
model.save(model_output_path)
print(f"Neural Network model saved to {model_output_path}")

np.save(x_train_path, X_train_num)
np.save(x_test_path, X_test_num)
np.save(x_train_path.replace('x_train_nn', 'x_train_seq_nn'), X_train_seq)
np.save(x_test_path.replace('x_test_nn', 'x_test_seq_nn'), X_test_seq)
np.save(y_test_path, y_test)
joblib.dump(scaler, 'app/data/preprocessed_data/scaler.pkl')

# Visualize Attention Weights (for some sample instances)
def plot_attention_weights(sample_idx, attention_weights, output_dir='app/int_images/attention_weights'):
    os.makedirs(output_dir, exist_ok=True)
    
    attention_map = attention_weights[sample_idx]
    attention_map = np.squeeze(attention_map)  # Remove unnecessary dimensions
    #attention_map = np.abs(attention_map)  # Ensure the weights are positive (if required)
    attention_map = tf.nn.softmax(attention_map, axis=-1).numpy()
    plt.figure(figsize=(12, 6))
    sns.heatmap(attention_map, cmap='viridis', xticklabels=False, yticklabels=False)
    
    plt.title(f'Attention Weights for Sequence Sample {sample_idx}')
    plt.xlabel('Time Steps')
    plt.ylabel('Attention Weight')
    
    attention_image_path = os.path.join(output_dir, f"attention_sample_{sample_idx}.png")
    plt.savefig(attention_image_path)
    plt.close()


plot_attention_weights(0, attention_weights)