import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from nnfs.datasets import spiral_data, sine_data
from sklearn.compose import ColumnTransformer
import os

def load_spiral_data(samples=100, classes=3):
    X, y = spiral_data(samples=samples, classes=classes)
    return X, y

def load_sine_data(samples=1000):
    X, y = sine_data(samples=samples)
    return X, y

def load_csv_dataset(filepath, target_column,categorical_columns, test_size=0.2, scale=True):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File '{filepath}' does not exist.")

    data = pd.read_csv(filepath)

    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")

    X = data.drop(columns=[target_column])
    y = data[target_column]

    if categorical_columns:
        categorical_transform = OneHotEncoder(drop='first')
        #Create the column transform
        preprocessor = ColumnTransformer(transformers=[('encoder',categorical_transform\
            ,categorical_columns)],remainder='passthrough')
        X = preprocessor.fit_transform(X)

    if y.dtype == 'object' or len(y.unique()) > 2:
        y = pd.get_dummies(y).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y.values, test_size=test_size)
    
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


def get_dataset(name, classes=None, validate=False):
    """Load one of the predefined datasets.

    Parameters
    ----------
    name : str
        ``"spiral"`` for the spiral dataset, ``"sine"`` for the sine wave
        dataset or ``"csv"`` for a CSV file.
    classes : int, optional
        Number of classes for the spiral dataset. If ``None`` the user will be
        asked for the value when ``name`` is ``"spiral"``.
    validate : bool, optional
        When ``True`` a smaller dataset suitable for validation is returned.
    """

    if name == 'spiral':
        if classes is None:
            classes = int(input("Enter the number of classes (2 or 3): ").strip())
        samples = 100 if validate else 1000
        return load_spiral_data(samples=samples, classes=classes)

    elif name == 'sine':
        samples = 100 if validate else 1000
        return load_sine_data(samples=samples)

    elif name == 'csv':
        filepath = input("Enter the path to the CSV file: ")
        target_column = input("Enter the target column name: ")
        categorical_column = input("Enter the categotical columns index separeated by comma: ")
        c_c = [int(x.strip()) for x in categorical_column.split(',') if x.strip()]
        if not filepath or not target_column:
            raise ValueError("Filepath and target_column must be provided for CSV datasets")

        return load_csv_dataset(filepath, target_column, c_c)
    else:
        raise ValueError("Unknown dataset. Options: 'spiral', 'sine', 'csv'")


# Example usage:
# X, y = get_dataset('spiral', classes=3)
