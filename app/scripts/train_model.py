import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import numpy as np

global_x_train = None
def train_and_save_model(preprocessed_data_path, model_output_path='app/models/random_forest_model.pkl', x_train_path = 'app/models/x_train.npy'):
    global global_x_train
    data = pd.read_csv(preprocessed_data_path)

    X = data.drop(columns=['Treatment_Response', 'Patient_ID'], axis = 1)
    y = data['Treatment_Response']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=250, random_state=42)
    model.fit(X_train, y_train)

    global_x_train = X_train

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy}")

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"Model saved to {model_output_path}")
    np.save(x_train_path, X_train)


preprocessed_file_path = 'app/data/preprocessed_data/preprocessed_clinical_data.csv'
#preprocessed_file_path2 = 'app/data/hiv_clinical_data.csv'
train_and_save_model(preprocessed_file_path)
