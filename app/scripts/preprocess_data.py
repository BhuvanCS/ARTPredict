import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import joblib
from Bio import SeqIO
import numpy as np
import helpers 


def preprocess_and_save(file_path, fasta_path, output_dir='app/data/preprocessed_data'):
    clinical_data = pd.read_csv(file_path)

    print("Initial clinical data info:")
    print(clinical_data.info())

    os.makedirs(output_dir, exist_ok=True)

    # Handling missing values for Treatment_Response
    imputer = SimpleImputer(strategy='most_frequent')
    clinical_data['Treatment_Response'] = imputer.fit_transform(clinical_data[['Treatment_Response']]).ravel()

    # Handling missing values for Viral_Load and CD4_Count
    mean_imputer = SimpleImputer(strategy='mean')
    clinical_data[['Viral_Load', 'CD4_Count']] = mean_imputer.fit_transform(clinical_data[['Viral_Load', 'CD4_Count']])

    # Parse FASTA file to extract sequence data and strain type
    sequence_data = {}
    strain_types = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        patient_id = record.id.split('|')[0].strip()
        strain_type = record.description.split('|')[1].strip()
        sequence_data[patient_id] = helpers.encode_sequence(str(record.seq))
        strain_types[patient_id] = strain_type

    print(sequence_data)
    clinical_data['Encoded_Sequence'] = clinical_data['Patient_ID'].map(lambda id: sequence_data.get(id, np.nan))
    clinical_data['Encoded_Sequence'] = clinical_data['Encoded_Sequence'].apply(lambda x: np.array(x).tolist() if x is not None else np.nan)
    clinical_data['HIV_Strain'] = clinical_data['Patient_ID'].map(strain_types)
    print(clinical_data['Encoded_Sequence'])

    # One-hot encode the HIV strain type
    strain_encoder = LabelEncoder()
    clinical_data['HIV_Strain'] = strain_encoder.fit_transform(clinical_data['HIV_Strain'])

    # Treatment_Response categorical to numerical
    label_encoder = LabelEncoder()
    clinical_data['Treatment_Response'] = label_encoder.fit_transform(clinical_data['Treatment_Response'])

    joblib.dump(imputer, os.path.join(output_dir, 'imputer.pkl'))
    joblib.dump(mean_imputer, os.path.join(output_dir, 'mean_imputer.pkl'))
    joblib.dump(strain_encoder, os.path.join(output_dir, 'strain_encoder.pkl'))
    joblib.dump(label_encoder, os.path.join(output_dir, 'label_encoder.pkl'))
    output_path = os.path.join(output_dir, 'preprocessed_clinical_data.csv')

    clinical_data.to_csv(output_path, index=False)
    print(f"Preprocessed clinical data saved to: {output_path}")

    return clinical_data


input_csv_path = 'app/data/hiv_clinical_data.csv' 
input_fasta_path = 'app/data/hiv_sequences.fasta' 
preprocessed_data = preprocess_and_save(input_csv_path, input_fasta_path)
