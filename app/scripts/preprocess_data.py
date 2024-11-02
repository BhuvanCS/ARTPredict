import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import joblib

def preprocess_and_save(file_path, output_dir='app/data/preprocessed_data'):
    clinical_data = pd.read_csv(file_path)

    print("Initial clinical data info:")
    print(clinical_data.info())

    os.makedirs(output_dir, exist_ok=True)

    # Handling missing values for Treatment_Response
    imputer = SimpleImputer(strategy='most_frequent')
    clinical_data['Treatment_Response'] = imputer.fit_transform(clinical_data[['Treatment_Response']]).ravel()

    # Handling missing values for Viral_Load and CD4_Count
    mean_imputer = SimpleImputer(strategy='mean')
    clinical_data['Viral_Load'] = mean_imputer.fit_transform(clinical_data[['Viral_Load', 'CD4_Count']])

    # Normalize continuous features (Viral Load, CD4 Count)
    scaler = StandardScaler()
    clinical_data[['Viral_Load', 'CD4_Count']] = scaler.fit_transform(clinical_data[['Viral_Load', 'CD4_Count']])

    # Treatment_Response categorical to numerical
    label_encoder = LabelEncoder()
    clinical_data['Treatment_Response'] = label_encoder.fit_transform(clinical_data['Treatment_Response'])

    joblib.dump(imputer, os.path.join(output_dir, 'imputer.pkl'))
    joblib.dump(mean_imputer, os.path.join(output_dir, 'mean_imputer.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    joblib.dump(label_encoder, os.path.join(output_dir, 'label_encoder.pkl'))
    output_path = os.path.join(output_dir, 'preprocessed_clinical_data.csv')

    clinical_data.to_csv(output_path, index=False)
    print(f"Preprocessed clinical data saved to: {output_path}")

    return clinical_data


input_csv_path = 'app/data/hiv_clinical_data.csv' 
preprocessed_data = preprocess_and_save(input_csv_path)
