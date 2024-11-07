import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import joblib
from Bio import SeqIO
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def preprocess_and_save(file_path, output_dir='app/data/preprocessed_data'):
    data = pd.read_csv(file_path)

        # Step 1: Encode categorical variables
    # One-hot encode 'strain_type' and 'treatment_response'
    


    # Handling missing values for Treatment_Response
    imputer = SimpleImputer(strategy='most_frequent')
    data['Treatment_Response'] = imputer.fit_transform(data[['Treatment_Response']]).ravel()

    strain_type_encoder = LabelEncoder()
    data['Strain_Type'] = strain_type_encoder.fit_transform(data['Strain_Type'])
    
    response_encoder = LabelEncoder()
    data['Treatment_Response'] = response_encoder.fit_transform(data['Treatment_Response'])

    # Handling missing values for Viral_Load and CD4_Count
    mean_imputer = SimpleImputer(strategy='mean')
    data[['Viral_Load', 'CD4_Count', 'Adherence_Level']] = mean_imputer.fit_transform(data[['Viral_Load', 'CD4_Count', 'Adherence_Level']])

    # Create a DataFrame for scaled numerical features
    

    # Step 3: Combine all features into a single DataFrame
    # Step 4: Encode sequence data (one-hot encoding or any other method)
    def one_hot_encode_sequence(sequence):
        # One-hot encode A, T, C, G (4 bases)
        encoding_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
        return np.array([encoding_dict.get(base, [0,0,0,0]) for base in sequence])

    # Apply one-hot encoding to sequence data
    sequence_encoded = np.array([one_hot_encode_sequence(seq) for seq in data['Sequence_Data']])
    sequence_encoded_reshaped = sequence_encoded.reshape(sequence_encoded.shape[0], -1)  # Flatten to 2D

    # Combine all features including sequence data
    
    sequence_columns = [f"Sequence_{i+1}" for i in range(sequence_encoded_reshaped.shape[1])]
    X_final_df = pd.DataFrame(sequence_encoded_reshaped, columns=sequence_columns)
    X_final_df['Patient_ID'] = data['Patient_ID']
    X_final_df['Viral_Load'] = data['Viral_Load']
    X_final_df['CD4_Count'] = data['CD4_Count']
    X_final_df['Adherence_Level'] = data['Adherence_Level']
    X_final_df['Strain_Type'] = data['Strain_Type']
    X_final_df['Treatment_Response'] = data['Treatment_Response']

    X_final_df = X_final_df[['Patient_ID', 'Viral_Load', 'CD4_Count', 'Adherence_Level', 'Strain_Type'] + sequence_columns + ['Treatment_Response']]


    joblib.dump(imputer, os.path.join(output_dir, 'imputer2.pkl'))
    joblib.dump(mean_imputer, os.path.join(output_dir, 'mean_imputer2.pkl'))
    joblib.dump(strain_type_encoder, os.path.join(output_dir, 'encoder_strain.pkl'))
    joblib.dump(response_encoder, os.path.join(output_dir, 'encoder_response.pkl'))
    output_path = os.path.join(output_dir, 'preprocessed_clinical_data_2.csv')

    os.makedirs(output_dir, exist_ok=True)
    X_final_df.to_csv(output_path, index=False)
    print(f"Preprocessed clinical data saved to: {output_path}")

    return X_final_df


input_csv_path = 'app/data/hiv_clinical_data_2.csv' 
 
preprocessed_data = preprocess_and_save(input_csv_path)



