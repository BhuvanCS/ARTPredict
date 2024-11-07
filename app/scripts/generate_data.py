import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate patient IDs
patient_ids = [f'P_{i+1}N' for i in range(300)]

# Generate adherence levels (0-1)
adherence_level = np.random.rand(300)

# Generate CD4 counts influenced by adherence level and some noise
cd4 = np.clip(np.random.normal(loc=500, scale=200, size=300) + (adherence_level * 100), 0, 1500)

# Generate viral load based on CD4 count and adherence level
viral_load = np.clip(2000000 / (cd4 / 15) * (1 - adherence_level * 0.7), 50, 1000000)

# Generate strain type based on viral load
strain_type = np.where(viral_load > 40000, 'HIV-1', 'HIV-2')

# Function to generate sequence data with mutations affecting treatment response
def generate_sequence(strain):
    # Basic sequence of length 100
    base_sequence = ''.join(np.random.choice(['A', 'T', 'C', 'G'], size=100))
    
    # Introduce mutations based on strain type
    if strain == 'HIV-1':
        # Introduce some mutations that might confer resistance
        mutation_positions = np.random.choice(range(100), size=5, replace=False)
        for pos in mutation_positions:
            base_sequence = base_sequence[:pos] + np.random.choice(['A', 'T', 'C', 'G']) + base_sequence[pos + 1:]
    else:
        # HIV-2 has fewer mutations affecting treatment response
        mutation_positions = np.random.choice(range(100), size=2, replace=False)
        for pos in mutation_positions:
            base_sequence = base_sequence[:pos] + np.random.choice(['A', 'T', 'C', 'G']) + base_sequence[pos + 1:]
    
    return base_sequence

sequence_data = [generate_sequence(st) for st in strain_type]

# Create treatment response based on a combination of factors including sequence data
treatment_response = []
for i in range(300):
    # Define logic for responders and non-responders based on viral load and adherence level
    if viral_load[i] < 20000 and adherence_level[i] > 0.8:  # High adherence and low viral load
        treatment_response.append('Responder')
    elif strain_type[i] == 'HIV-1' and adherence_level[i] < 0.5:  # Low adherence with HIV-1 strain
        treatment_response.append('Non-Responder')
    elif 'A' in sequence_data[i][:10] and adherence_level[i] > 0.7:  # Specific sequence pattern indicating sensitivity
        treatment_response.append('Responder')
    else:
        treatment_response.append('Non-Responder')

# Create DataFrame
data = pd.DataFrame({
    'Patient_ID': patient_ids,
    'Viral_Load': viral_load,
    'CD4_Count': cd4,
    'Adherence_Level': adherence_level,
    'Strain_Type': strain_type,
    'Sequence_Data': sequence_data,
    'Treatment_Response': treatment_response
})

# Save to CSV (optional)
data.to_csv('app/data/hiv_clinical_data_2.csv', index=False)
