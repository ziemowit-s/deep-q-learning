import random

import numpy as np
import tensorflow as tf

from agents.rnn_layernorm_agent.rnn_layernorm_core import RNNLayernormCore
from utils import MemUnit


class RNNAgent(object):

    def __init__(self, input_length, input_size, output_size, action_size, activation='relu',
                 hidden_units=16, gamma=0.9, learning_rate=0.6, batch_size=128, **kwargs):
        self.input_length = input_length
        self.input_size = input_size
        self.batch_size = batch_size
        self.model = RNNLayernormCore(input_length, input_size, output_size, activation,
                                      hidden_units=hidden_units, batch_size=batch_size)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.action_size = action_size
        self.memory = []
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def _get_memory(self):
        if len(self.memory) < self.batch_size:
            return self.memory
        else:
            return random.sample(self.memory, self.batch_size)

    def _reshape_state(self, state):
        return np.reshape(state, [1, self.input_length, self.input_size])

    def _predict(self, state):
        state = np.reshape(state, [1, self.input_length, self.input_size])
        return self.model.predict(self.sess, state)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self._predict(state))

    def learn(self):
        memory_sample = self._get_memory()

        for mem_unit in memory_sample:
            target = mem_unit.reward
            if mem_unit.is_done:
                break

            next_state_prediction = self._predict(mem_unit.next_state)[0][0]
            current_state = self._reshape_state(mem_unit.state)
            target = target + self.gamma * next_state_prediction
            target = np.array([target])

            output = self.model.step(self.sess, current_state, target, self.learning_rate)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def remember(self, state, action, reward, next_state, done):
        mem_unit = MemUnit(state, action, reward, next_state, done)
        self.memory.append(mem_unit)

    def save(self, path_prefix):
        self.model.save(self.sess, path_prefix)

    def load(self, path_prefix):
        self.model.load(self.sess, path_prefix)

    def __del__(self):
        self.sess.close()