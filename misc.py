import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
import joblib

def load_data():
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    feature_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target
    return df

def preprocess(df, target_col='MEDV', scale=False):
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].values
    scaler = None
    if scale:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X, y, scaler

def split_data(X, y, test_size=0.2, random_state=None):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return mse

def run_repeated_experiment(model, df, n_repeats=5, test_size=0.2, random_states=None, scale=False):
    if random_states is None:
        random_states = list(range(n_repeats))
    mses = []
    for rs in random_states:
        X, y, _ = preprocess(df, scale=scale)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=rs)
        cloned = clone(model)
        train_model(cloned, X_train, y_train)
        mse = evaluate_model(cloned, X_test, y_test)
        mses.append(mse)
    return float(np.mean(mses)), mses

def save_model(model, filepath):
    joblib.dump(model, filepath)
