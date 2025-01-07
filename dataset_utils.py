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

def get_dataset(name,classes,validate=False):
    if name == 'spiral' and classes == 2:
        if validate==False:
            return load_spiral_data(samples=1000, classes=2)
        elif validate==True:
            return load_spiral_data(samples=100, classes=2)

    elif name == 'spiral' and classes ==3:
        if validate==False:
            return load_spiral_data(samples=1000, classes=3)
        elif validate==True:
            return load_spiral_data(samples=100, classes=3)

    elif name == 'sine':
        if validate==False:
            return load_sine_data(samples=1000)
        elif validate==True:
            return load_sine_data(samples=100)

    elif name == 'csv':
        filepath = (input("Enter the path to the CSV file: "))
        target_column = (input("Enter the target column name: "))
        categorical_column = (input("Enter the categotical columns index separeated by comma: "))
        c_c = [int(x.strip()) for x in categorical_column.split(",") if x.strip()]
        if not filepath or not target_column:
            raise ValueError("Filepath and target_column must be provided for CSV datasets")
        
        return load_csv_dataset(filepath, target_column,c_c)
    else:
        raise ValueError("Unknown dataset. Options: 'spiral', 'sine', 'csv'")



#print(os.getcwd())
#X_train, X_test, y_train, y_test = get_dataset(name)