import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml


def load_data(path):
    data = pd.read_csv(path)
    return data


def process_and_split_data(data, cols_to_encode, target_name, test_size, random_seed):
    encoded_data = pd.get_dummies(data, columns=cols_to_encode, drop_first=True)

    X = encoded_data.drop(columns=[target_name])
    y = encoded_data[target_name]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_seed
    )

    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    return train_data, test_data


def save_data(train, test, train_name, test_name):
    train.to_csv(train_name, index=False)
    test.to_csv(test_name, index=False)


if __name__ == "__main__":
    # Extract parameters from yaml file
    params = yaml.safe_load(open("params.yaml"))
    split_params = params["split"]
    encoding_params = params["encoding"]
    target_params = params["target"]

    dataset_path = split_params["dataset_path"]
    cols_to_encode = encoding_params["columns"]
    target_name = target_params["name"]
    test_size = split_params["test_size"]
    random_seed = split_params["random_seed"]

    data = load_data(dataset_path)
    train_dataset, test_dataset = process_and_split_data(data, cols_to_encode, target_name, test_size, random_seed)
    save_data(train_dataset, test_dataset, 'data/processed_train_data.csv', 'data/processed_test_data.csv')
