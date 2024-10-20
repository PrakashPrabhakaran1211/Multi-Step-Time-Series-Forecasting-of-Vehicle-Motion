import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np


def my_lstm_model(xTrain, xTest, yTrain, yTest):
    n_input = np.shape(xTrain)[1] * np.shape(xTrain)[2]
    xTrain = np.reshape(xTrain, (np.shape(xTrain)[0], 1, n_input))
    n_output = np.shape(yTrain)[1] * np.shape(yTrain)[2]
    yTrain = np.reshape(yTrain, (np.shape(yTrain)[0], n_output))
    xTest = np.reshape(xTest, (np.shape(xTest)[0], 1, n_input))
    yTest = np.reshape(yTest, (np.shape(yTest)[0], n_output))

    model = Sequential()
    model.add(LSTM(50,activation='relu', input_shape=(np.shape(xTrain)[1], np.shape(xTrain)[2])))

    model.add(Dense(n_output))
    model.compile(loss='mse', optimizer='Adam')

    # fit network
    history = model.fit(xTrain, yTrain, epochs=20, batch_size=50, validation_data=(xTest, yTest), verbose=1,
                        shuffle=False)

    # save the trained model
    filepath = 'trained_models/LSTM_model.h5'
    model.save(filepath)

    # load trained model
    model = keras.models.load_model('trained_models/LSTM_model.h5')

    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig('loss_visualization/loss_comparison_LSTM.pdf')

    plt.show()

    return model
