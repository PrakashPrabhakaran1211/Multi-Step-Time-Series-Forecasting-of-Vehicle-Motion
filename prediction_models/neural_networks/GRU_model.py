import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, LSTM, GRU
import numpy as np

def GRU_model(xTrain, xTest, yTrain, yTest):
    
    # flatten the input and output
    n_input = np.shape(xTrain)[1] * np.shape(xTrain)[2]
    xTrain = np.reshape(xTrain, (np.shape(xTrain)[0], 1, n_input))
    n_output = np.shape(yTrain)[1] * np.shape(yTrain)[2]
    yTrain = np.reshape(yTrain, (np.shape(yTrain)[0], n_output))

    xTest = np.reshape(xTest, (np.shape(xTest)[0], 1, n_input))
    yTest = np.reshape(yTest, (np.shape(yTest)[0], n_output))

    # define GRU model
    model = Sequential()
    model.add(GRU(30, input_shape=(np.shape(xTrain)[1],np.shape(xTrain)[2]), activation='relu', return_sequences=True))
    model.add(GRU(25, activation='relu', return_sequences=True))
    model.add(GRU(20, activation='relu', return_sequences=False))
    model.add(Dense(n_output))
    model.compile(loss='mse', optimizer='Adam', metrics='accuracy')
    
    # fit the keras model on the dataset
    history =model.fit(xTrain, yTrain, epochs=20, batch_size=100, verbose=1, validation_data=(xTest, yTest))
    
    # save the trained model
    filepath = 'trained_models/GRU_model.h5'
    model.save(filepath)
    
     # load trained model
    model = keras.models.load_model('trained_models/GRU_model.h5')

    # Training and validation loss plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig('loss_visualization/loss_comparison_GRU.pdf')

    return model