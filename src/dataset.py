from config import Config
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, Normalizer
import pandas as pd
import numpy as np
import argparse

config = Config()


def one_hot_encode(df, cols):
    encoder = OneHotEncoder()
    encoded_features = encoder.fit_transform(df[cols]).toarray()
    feature_names = encoder.get_feature_names_out(cols)
    df[feature_names] = encoded_features
    df = df.drop(cols, axis=1)
    return df


def standard_encode(df, cols):
    for col in cols:
        df[col], _ = pd.factorize(df[col])
    return df


def extract_features(dataset, drop_cols, label_col):
    df = pd.read_csv(dataset)

    # Drop empty rows
    df = df.dropna()
    # Drop ID row
    df = df.drop(drop_cols, axis=1)

    # Select categorical columns
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    # Exclude the label column
    categorical_cols = [col for col in categorical_cols if col != label_col]

    # Categorical features encoding
    if config.ENCODING == 'one-hot':
        df = one_hot_encode(df, categorical_cols)
    elif config.ENCODING == 'standard':
        df = standard_encode(df, categorical_cols)

    # Label numerical classification
    if (df[label_col].dtype == 'object'):
        df[label_col], _ = pd.factorize(df[label_col])

    # Extract labels (y)
    y = df[label_col]

    # Extract feature matrix (X)
    X = df.drop([label_col], axis=1)

    # Normalization
    scaler = StandardScaler()
    scaler.fit(X)
    X_norm = scaler.transform(X)

    # Test and train split
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=0.2, random_state=42)

    # Convert to ndarray
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return X_train, X_test, y_train, y_test


def save_to_hard_drive(features):
    joblib.dump(features, config.FEATURES_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # -d flag arguments
    parser.add_argument('-d', '--drop', type=str,
                        help='usless columns to drop. separated by ,')
    # -l flag arguments
    parser.add_argument('-l', '--label', type=str,
                        help='target label column (y)')
    # Add a positional argument for the input file
    parser.add_argument('dataset', type=str, help='dataset csv file')

    args = parser.parse_args()

    drop_cols = args.drop.split(',')
    label_col = args.label
    dataset = args.dataset

    features = extract_features(dataset, drop_cols, label_col)

    save_to_hard_drive(features)

    print(f"Successfully extracted features at {config.FEATURES_PATH}")
