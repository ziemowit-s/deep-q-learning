import random
from collections import deque

import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from agents.keras_dense_agent.keras_dense import KerasDense


class KerasAgent(object):
    def __init__(self, state_size, action_size, activation='relu',
                 hidden_units=24, gamma=0.95, learning_rate=0.001, batch_size=32, **kwargs):
        self.state_size = state_size
        self.batch_size = batch_size
        self.model = KerasDense(state_size, action_size)

        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def act(self, state):
        state = np.reshape(state, [1, 4])
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.model.model.predict(state)[0])

    def learn(self):
        memory_sample = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in memory_sample:
            next_state = np.reshape(next_state, [1, 4])
            state = np.reshape(state, [1, 4])

            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.model.predict(next_state)[0]))
            target_f = self.model.model.predict(state)
            target_f[0][action] = target
            self.model.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def remember(self, state, action, reward, next_state, done):
        mem_unit = (state, action, reward, next_state, done)
        self.memory.append(mem_unit)
