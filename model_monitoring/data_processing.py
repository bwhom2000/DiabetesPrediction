import pandas as pd

smoking_history_mapping = {
    'No Info': 'non_smoker',
    'never': 'non_smoker',
    'former': 'former_smoker',
    'not current': 'former_smoker',
    'ever': 'former_smoker',
    'current': 'current_smoker'
}

expected_features = [
    "age",
    "hypertension",
    "heart_disease",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
    "gender_Female",
    "gender_Male",
    "gender_Other",
    "smoking_history_current_smoker",
    "smoking_history_former_smoker",
    "smoking_history_non_smoker",
]


def preprocess_data(data):

    input_df = data

    # Map the smoking history using the defined mapping
    input_df['smoking_history'] = input_df['smoking_history'].map(smoking_history_mapping)

    # Create boolean columns for gender
    input_df['gender_Female'] = input_df['gender'] == 'Female'
    input_df['gender_Male'] = input_df['gender'] == 'Male'
    input_df['gender_Other'] = input_df['gender'] == 'Other'

    # Create boolean columns for smoking history
    input_df['smoking_history_current_smoker'] = input_df['smoking_history'] == 'current_smoker'
    input_df['smoking_history_former_smoker'] = input_df['smoking_history'] == 'former_smoker'
    input_df['smoking_history_non_smoker'] = input_df['smoking_history'] == 'non_smoker'

    # Drop original columns
    input_df.drop(columns=['gender', 'smoking_history'], inplace=True)

    # Ensure all expected features are present
    for feature in expected_features:
        if feature not in input_df.columns:
            input_df[feature] = 0  # Assign 0 if the feature is missing

    # Reorder the DataFrame to match expected feature order
    input_df = input_df[expected_features]

    return input_df  # Return as a numpy array
