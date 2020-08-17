from time import strftime, gmtime
import tensorflow as tf
import numpy as np
import os


class FastWeightsCore(object):

    def __init__(self, input_length, input_size, output_size, activation='relu',
                 fastweight_past_learning_rate=0.8, hidden_units=16, inner_loop_steps=1, gain=1):
        super(FastWeightsCore, self).__init__()
        if isinstance(hidden_units, list):
            hidden_units = hidden_units[0]

        date = strftime("%Y-%m-%d-%M-%S", gmtime())
        self.modelname = "%s-%s_hidden_units-%s" % (self.__class__.__name__, hidden_units, date)

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
            x_ini = tf.sqrt(float(input_size))
            w_x = tf.Variable(tf.random_uniform([input_size, hidden_units], -x_ini, x_ini))
            b_x = tf.Variable(tf.random_uniform([hidden_units], -x_ini, x_ini))

            # hidden weights
            w_h = tf.Variable(np.identity(hidden_units, dtype=float), dtype=tf.float32)

            # layernorm bias
            b_norm = tf.Variable(tf.random_uniform([hidden_units], -x_ini, x_ini))

            # dense
            w_d = tf.Variable(np.identity(hidden_units, dtype=float), dtype=tf.float32)
            b_d = tf.Variable(tf.random_uniform([hidden_units], -x_ini, x_ini))

            # output weights
            soft_ini = tf.sqrt(float(hidden_units))
            w_softmax = tf.Variable(tf.random_uniform([hidden_units, output_size],
                                                      -soft_ini, soft_ini))
            b_softmax = tf.Variable(tf.random_uniform([output_size], -soft_ini, soft_ini))

        # Vectors for initial computation
        h = tf.zeros([batch_size, hidden_units], dtype=tf.float32, name='h')
        a = tf.zeros([batch_size, hidden_units, hidden_units], name='a')
        h_logit = tf.zeros([batch_size, hidden_units], name='last_h_logit')

        # Output variables
        self.train_step = None
        self.loss = None
        self.accuracy = None

        # RNN loop
        for rnn_i in range(input_length):
            # H0
            last_h_logit = h_logit
            h_logit = tf.matmul(self.x[:, rnn_i, :], w_x) + b_x + tf.matmul(h, w_h)
            h = self._activation(h_logit, name='rnn_activation')
            # A
            a = a * self.lambda_decay + tf.matmul(tf.transpose(h), h) * fastweight_past_learning_rate

            # on t=0 continue
            if rnn_i == 0:
                continue

            # Fast Weights inner loop
            hs = h
            for fw_i in range(0, inner_loop_steps):
                hs = tf.reshape(h, [batch_size, hidden_units, 1])
                a_hs = tf.reshape(tf.matmul(a, hs), [batch_size, hidden_units], name='a_hs')
                hs = tf.add(last_h_logit, a_hs)

                hs = self._layernorm(hs, batch_size, self.gain, b_norm)
                hs = self._activation(hs, name='fast_weight_activation')

            h = hs

        h_dense = tf.nn.relu((tf.matmul(h, w_d) + b_d))

        # Softmax logit
        self.softmax_logit = tf.matmul(h_dense, w_softmax) + b_softmax

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
            tf.nn.softmax_cross_entropy_with_logits(logits=self.softmax_logit, labels=self.y))

        # Gradient
        self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        # Accuracy
        correct = tf.equal(tf.argmax(self.softmax_logit, axis=1), tf.argmax(self.y, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    def save(self, sess, epoch_num, batch_num, folder_prefix):
        date = strftime("%Y-%m-%d-%M-%S", gmtime())
        foldername = "%s/%s/%s-epoch_%s-batch_%s/" % \
                     (folder_prefix, self.modelname, date, epoch_num, batch_num)

        if not os.path.exists(foldername):
            os.makedirs(foldername)

        self.saver.save(sess, foldername)

    def load(self, sess, folder_prefix):
        self.saver.restore(sess, folder_prefix)

    def step(self, sess, batch_x, batch_y, learning_rate, lambda_decay):
        input_dict = {self.x: batch_x, self.y: batch_y,
                      self.learning_rate: learning_rate, self.lambda_decay: lambda_decay}

        return sess.run([self.loss, self.accuracy, self.train_step], input_dict)

    def evaluate(self, sess, x, y, learning_rate, lambda_decay):
        input_dict = {self.x: x, self.y: y,
                      self.learning_rate: learning_rate, self.lambda_decay: lambda_decay}
        return sess.run([self.loss, self.accuracy], input_dict)
