from keras.models import Sequential
from keras.layers import Dense, LSTM, Input, Dropout, Reshape
from keras import optimizers


def load_model(features_dim):
    model = Sequential()

    model.add(Input(shape=(features_dim, )))
    model.add(Reshape((features_dim, 1)))

    model.add(LSTM(32, activation='relu'))
    model.add(Dropout(0.8))

    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy',
                    optimizer=optimizers.Adam(learning_rate=10e-4), metrics=['accuracy'])

    return model
