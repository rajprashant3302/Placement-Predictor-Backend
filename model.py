import sys, json, pickle
import numpy as np
import pandas as pd

# Load trained models
with open('random_forest_model.pkl', 'rb') as f:
    RF_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('OHE.pkl', 'rb') as f:
    encoder = pickle.load(f)

# Get features from Node.js
features = json.loads(sys.argv[1])
features = np.array(features).reshape(1, -1)

# Columns must match training dataset
columns = ['CGPA', 'Internships', 'Projects', 'CompetitiveRank', 'Branch',
           'CodeforcesRating', 'CommunicationSkill', 'ExperienceMonths', 'Age',
           'CollegeTag', 'Grade10', 'Grade12', 'Backlogs', 'Gender']

df_in = pd.DataFrame(features, columns=columns)

# Separate categorical and numeric
cat_cols = ['Branch', 'CollegeTag', 'Gender']
numeric_cols = ['CGPA', 'Internships', 'Projects', 'CompetitiveRank',
                'CodeforcesRating', 'CommunicationSkill', 'ExperienceMonths',
                'Age', 'Grade10', 'Grade12', 'Backlogs']

# Encode categorical features
encoded_input = encoder.transform(df_in[cat_cols])
encoded_input_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(cat_cols))

# Combine numeric and encoded categorical features
input_combined = pd.concat([df_in[numeric_cols], encoded_input_df], axis=1)

# Scale features
input_scaled = scaler.transform(input_combined)

# Predict using Random Forest model
predicted_package = RF_model.predict(input_scaled)

# Send prediction back to Node.js
print(predicted_package[0])
