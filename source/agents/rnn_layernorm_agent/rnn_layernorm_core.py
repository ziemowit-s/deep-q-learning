import random
from time import strftime, gmtime

import tensorflow as tf
import numpy as np
import os

from utils import MemUnit


class RNNLayernormCore(object):

    def __init__(self, input_length, input_size, output_size, activation='relu',
                 hidden_units=16, gain=1, batch_size=128):
        if isinstance(hidden_units, list):
            hidden_units = hidden_units[0]

        date = strftime("%Y-%m-%d-%M-%S", gmtime())
        self.modelname = "%s-%s_hidden_units-%s" % (self.__class__.__name__, hidden_units, date)

        self.batch_size = batch_size
        self.input_lenth = input_length
        self.input_size = input_size

        # set activation
        if activation == 'relu':
            self._activation = tf.nn.relu
        elif activation == 'sigmoid':
            self._activation = tf.nn.sigmoid
        elif activation == 'tanh':
            self._activation = tf.nn.tanh

        with tf.variable_scope("placeholders"):
            # x, y placeholders
            self.x = tf.placeholder(tf.float32, shape=[None, input_length, input_size], name='x')
            self.y = tf.placeholder(tf.float32, shape=[None, output_size], name='y')
            batch_size = tf.shape(self.x)[0]

            # paramethers placeholders
            self.gain = gain * tf.ones([hidden_units], name="gain")
            self.learning_rate = tf.placeholder(tf.float32, [], name="learning_rate")
            self.lambda_decay = tf.placeholder(tf.float32, [], name="lambda_decay")

        with tf.variable_scope("weights"):
            # input weights
            x_ini = tf.sqrt(1.0 / input_size)
            w_x = tf.Variable(tf.random_uniform([input_size, hidden_units], -x_ini, x_ini))
            b_x = tf.Variable(tf.random_uniform([hidden_units], -x_ini, x_ini))

            # hidden weights
            w_h = tf.Variable(np.identity(hidden_units, dtype=float), dtype=tf.float32)

            # layernorm bias
            b_norm = tf.Variable(tf.random_uniform([hidden_units], -x_ini, x_ini))

            # output weights
            soft_ini = tf.sqrt(1.0 / hidden_units)
            w_softmax = tf.Variable(tf.random_uniform([hidden_units, output_size],
                                                      -soft_ini, soft_ini))
            b_softmax = tf.Variable(tf.random_uniform([output_size],
                                                      -soft_ini, soft_ini))

        # Vectors for initial computation
        h = tf.zeros([batch_size, hidden_units], dtype=tf.float32, name='h')

        # Output variables
        self.train_step = None
        self.loss = None
        self.accuracy = None

        # RNN loop
        for rnn_i in range(input_length):
            h_logit = tf.matmul(self.x[:, rnn_i, :], w_x, a_is_sparse=True) + b_x + tf.matmul(h, w_h)
            h_logit = self._layernorm(h_logit, batch_size, self.gain, b_norm)
            h = self._activation(h_logit, name='rnn_activation')

        # Softmax logit
        self.softmax_logit = tf.matmul(h, w_softmax) + b_softmax

        self.saver = tf.train.Saver()

        self._compile()

    @staticmethod
    def _layernorm(layer, batch_size, gain, bias):
        mean_h = tf.reshape(tf.reduce_mean(layer, reduction_indices=1), [batch_size, 1],
                            name='mean_norm')
        sigma = tf.sqrt(tf.reduce_mean(tf.square(layer - mean_h), reduction_indices=1))
        sigma = tf.reshape(sigma, [batch_size, 1], name='sigma')

        nominator = tf.multiply(gain, (layer - mean_h))
        return tf.divide(nominator, sigma, name='div_norm') + bias

    def _compile(self):
        # Loss
        self.loss = tf.reduce_mean(
            tf.losses.mean_squared_error(predictions=self.softmax_logit, labels=self.y))
        # Gradient
        self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

    def save(self, sess, epoch_num, batch_num, folder_prefix):
        date = strftime("%Y-%m-%d-%M-%S", gmtime())
        foldername = "%s/%s/%s-epoch_%s-batch_%s/" % \
                     (folder_prefix, self.modelname, date, epoch_num, batch_num)

        if not os.path.exists(foldername):
            os.makedirs(foldername)

        self.saver.save(sess, foldername)

    def load(self, sess, folder_prefix):
        self.saver.restore(sess, folder_prefix)

    def step(self, sess, batch_x, batch_y, learning_rate):
        input_dict = {self.x: batch_x, self.y: batch_y, self.learning_rate: learning_rate}
        return sess.run([self.loss, self.train_step], input_dict)

    def evaluate(self, sess, x, y):
        input_dict = {self.x: x, self.y: y}
        return sess.run([self.loss], input_dict)

    def predict(self, sess, x):
        input_dict = {self.x: x}
        return sess.run([self.softmax_logit], input_dict)