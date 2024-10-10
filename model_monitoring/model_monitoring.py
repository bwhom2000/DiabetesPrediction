from data_processing import preprocess_data

import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.metrics import f1_score

def load_data(reference_path, test_path):
    """Load reference and test datasets."""
    reference = pd.read_csv(reference_path)
    test = pd.read_csv(test_path)
    return reference, test

def load_model(model_path):
    """Load the trained model."""
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model


def make_predictions(reference, test, model):
    """Make predictions on reference and test datasets."""
    # Create copies of the DataFrames to avoid modifying the original ones
    reference_copy = reference.copy()
    test_copy = test.copy()

    # Preprocess the features before predictions
    reference_data_feats = preprocess_data(reference_copy)
    test_data_feats = preprocess_data(test_copy)

    # Make predictions
    reference['predicted'] = model.predict(reference_data_feats)
    test['predicted'] = model.predict(test_data_feats)

    # Rename the target column for clarity
    reference.rename(columns={'diabetes': 'target'}, inplace=True)
    test.rename(columns={'diabetes': 'target'}, inplace=True)

    # Convert columns to object type
    columns_to_convert = ["hypertension", "heart_disease", "target", "predicted"]
    reference[columns_to_convert] = reference[columns_to_convert].astype('object')
    test[columns_to_convert] = test[columns_to_convert].astype('object')

    return reference, test

def create_directories():
    """Create directories for saving plots and results."""
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)

def visualize_distributions(reference, test):
    """Visualize distributions of features and save plots."""
    features = [
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "smoking_history",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level",
        "target",
        "predicted"
    ]

    num_features = len(features)
    fig, axes = plt.subplots(nrows=num_features, ncols=1, figsize=(10, num_features * 4))

    for ax, feature in zip(axes, features):
        if reference[feature].dtype in ['float64', 'int64']:
            sns.kdeplot(reference[feature], ax=ax, label='Reference', color='blue', fill=True, alpha=0.5)
            sns.kdeplot(test[feature], ax=ax, label='Test', color='orange', fill=True, alpha=0.5)
            ax.set_title(f'Distribution of {feature}')
            ax.set_xlabel(feature)
            ax.set_ylabel('Density')
        else:
            reference_counts = reference[feature].value_counts().reset_index()
            reference_counts.columns = [feature, 'count']
            reference_counts['dataset'] = 'Reference'

            test_counts = test[feature].value_counts().reset_index()
            test_counts.columns = [feature, 'count']
            test_counts['dataset'] = 'Test'

            combined_counts = pd.concat([reference_counts, test_counts])

            sns.barplot(x=feature, y='count', hue='dataset', data=combined_counts, ax=ax, palette={'Reference': 'blue', 'Test': 'orange'}, alpha=0.7)
            ax.set_title(f'Distribution of {feature}')
            ax.set_xlabel(feature)
            ax.set_ylabel('Count')

        ax.legend(title='Dataset')

    plt.tight_layout()
    plt.savefig('plots/distributions.png')  # Save the plot
    plt.close()

def drift_analysis(reference, test, feature):
    """Perform KS test for drift analysis on a feature."""
    ks_statistic, p_value = ks_2samp(reference[feature], test[feature])
    drift_detected = p_value < 0.05
    return {
        'KS Statistic': ks_statistic,
        'p-value': p_value,
        'Drift Detected': drift_detected
    }

def perform_drift_analysis(reference, test):
    """Perform drift analysis for multiple features."""
    features = [
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "smoking_history",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level",
        "target",
        "predicted"
    ]

    results = {feature: drift_analysis(reference, test, feature) for feature in features}
    drift_results_df = pd.DataFrame(results).T
    drift_results_df.to_csv('results/drift_analysis_results.csv')

def calculate_f1_scores(reference, test):
    """Calculate F1 scores for test and reference datasets."""
    f1_test = f1_score(test['target'].astype(int), test['predicted'].astype(int))
    f1_reference = f1_score(reference['target'].astype(int), reference['predicted'].astype(int))
    return f1_test, f1_reference

def save_performance_metrics(f1_test, f1_reference):
    """Save performance metrics to CSV."""
    performance_df = pd.DataFrame({
        'Metric': ['F1 Score (Test)', 'F1 Score (Reference)'],
        'Score': [f1_test, f1_reference]
    })
    performance_df.to_csv('results/performance_metrics.csv', index=False)

def main():
    # Define paths
    reference_path = '../data/raw_data.csv'
    test_path = '../data/new_data_target.csv'
    model_path = 'model.pkl'

    # Load data and model
    reference, test = load_data(reference_path, test_path)
    model = load_model(model_path)

    # Create directories
    create_directories()

    # Make predictions
    reference, test = make_predictions(reference, test, model)

    # Visualize distributions
    visualize_distributions(reference, test)

    # Perform drift analysis
    perform_drift_analysis(reference, test)

    # Calculate F1 scores
    f1_test, f1_reference = calculate_f1_scores(reference, test)

    # Save performance metrics
    save_performance_metrics(f1_test, f1_reference)

    print("Results and plots saved successfully.")

if __name__ == "__main__":
    main()
