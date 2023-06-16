from model import load_model
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from config import Config
from plot import plot_history
from sklearn.utils.class_weight import compute_class_weight

config = Config()


def train(model, X_train, X_test, y_train, y_test):
    class_weights = compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    # Train the model
    X_train = np.expand_dims(X_train, axis=2)
    history = model.fit(X_train, y_train, batch_size=config.BATCH_SIZE,
                        epochs=config.EPOCHS, validation_data=(X_test, y_test), class_weight=class_weight_dict)

    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)

    # Print the prediction results
    unique_arr, counts = np.unique(y_pred, return_counts=True)
    for i in range(len(unique_arr)):
        print(f'{unique_arr[i]} occurs {counts[i]} times')

    # Test set loss and accuracy
    test_loss, test_acc = model.evaluate(X_test, y_test)

    print(f'Test Accuracy:  {test_acc:.3f}')
    print(f'Test Loss:      {test_loss:.3f}')

    # Evaluation metrics
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f'F1 score:       {f1:.3f}')
    print(f'Accuracy:       {accuracy:.3f}')
    print(f'Precision:      {precision:.3f}')
    print(f'Recall:         {recall:.3f}')

    # Save the model in HD
    model.save(config.MODEL_PATH)
    print(f"Successfully saved the model at {config.MODEL_PATH}")

    return history


if __name__ == "__main__":
    # Load the features from HD
    features = joblib.load(config.FEATURES_PATH)

    # Load the model
    model = load_model(features[0].shape[1])

    # Train the model
    history = train(model, *features)

    # Visualize the train history
    plot_history(history)
