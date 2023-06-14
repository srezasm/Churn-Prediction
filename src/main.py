import argparse
from dataset import extract_features
from model import load_model
from train import train
from plot import plot_history


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

    # Extract and save the features in HD
    X_train, X_test, y_train, y_test = extract_features(dataset, drop_cols, label_col)

    # Load the model
    model = load_model(X_train.shape[1])

    # Train the model
    history = train(model, X_train, X_test, y_train, y_test)

    # Plot the train history
    plot_history(history)
