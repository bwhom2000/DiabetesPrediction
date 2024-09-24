import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    """
    Load data from a CSV file.

    Parameters:
    path (str): The file path to the CSV file.

    Returns:
    pd.DataFrame: A DataFrame containing the loaded data.
    """
    return pd.read_csv(path)

def process_data(data, cols_to_encode):
    """
    Process the input DataFrame by encoding specified categorical columns.

    Parameters:
    data (pd.DataFrame): The input DataFrame to be processed.
    cols_to_encode (list): List of column names to be one-hot encoded.

    Returns:
    pd.DataFrame: A DataFrame with one-hot encoded columns.
    """
    # Perform one-hot encoding on specified categorical columns
    return pd.get_dummies(data, columns=cols_to_encode, drop_first=True)

def split_data(data, target_name, test_size, random_seed):
    """
    Split the data into training and testing sets.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing features and target.
    target_name (str): The name of the target column.
    test_size (float): The proportion of the dataset to include in the test split.
    random_seed (int): Random seed for reproducibility.

    Returns:
    tuple: A tuple containing the training and testing sets (X_train, X_test, y_train, y_test).
    """
    X = data.drop(columns=[target_name])
    y = data[target_name]

    # Split the data into training and testing sets
    return train_test_split(X, y, test_size=test_size, random_state=random_seed)
