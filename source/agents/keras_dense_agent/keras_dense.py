from time import strftime, gmtime
import tensorflow as tf
import os

from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class KerasDense(object):

    def __init__(self, input_size, output_size):
        self.model = Sequential()
        self.model.add(Dense(100, input_dim=input_size, activation='relu'))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(output_size, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=0.001))

    def step(self, current_state, target_f, epochs=1, verbose=0):
        self.model.fit(current_state, target_f, epochs, verbose)

    def predict(self, x):
        return self.model.predict(x, verbose=False)