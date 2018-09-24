import logging
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages
from equation import Equation
from config import Config


TF_DTYPE = tf.float64
MOMENTUM = 0.99
EPSILON = 1e-6
DELTA_CLIP = 50.0


xrange = range  # python 2.0 migrated to 3.0


def rsum(x):
    return tf.reduce_sum(x, 1, keepdims=True)


class FeedForwardModel(object):
    """The fully connected neural network model."""
    def __init__(self, config: Config, bsde: Equation, sess: tf.Session):
        self._config = config
        self._bsde = bsde
        self._sess = sess
        # make sure consistent with FBSDE equation
        self._dim = bsde.dim
        self._num_time_interval = bsde.num_time_interval
        self._total_time = bsde.total_time
        # ops for statistics update of batch normalization
        self._extra_train_ops = []

        # Below are set up by build()
        self._dw = None
        self._x = None
        self._is_training = None
        self._y_init = None
        self._loss = None
        self._train_ops = None
        self._t_build = None  # time to build

        self._loss_trained = None
        self._y0_trained = None

    def train(self):
        start_time = time.time()
        # to save iteration results
        training_history = []
        # for validation
        dw_valid, x_valid = self._bsde.sample(self._config.valid_size)
        # can still use batch norm of samples in the validation phase
        feed_dict_valid = {self._dw: dw_valid, self._x: x_valid, self._is_training: False}
        # initialization
        self._sess.run(tf.global_variables_initializer())
        self._loss_trained = None
        self._y0_trained = None

        # begin training iteration
        for step in xrange(self._config.num_iterations+1):
            if step % self._config.logging_frequency == 0:
                loss, y0 = self._sess.run([self._loss, self._y_init], feed_dict=feed_dict_valid)
                elapsed_time = time.time()-start_time+self._t_build
                training_history.append([step, loss, y0, elapsed_time])
                if self._config.verbose:
                    logging.info("step: %5u,    loss: %.4e,   Y0: %.4e,  elapsed time %3u" % (
                        step, loss, y0, elapsed_time))
                self._loss_trained = loss
                self._y0_trained = y0
            dw_train, x_train = self._bsde.sample(self._config.batch_size)
            self._sess.run(self._train_ops, feed_dict={self._dw: dw_train,
                                                       self._x: x_train,
                                                       self._is_training: True})
        return np.array(training_history)

    def build(self):
        start_time = time.time()
        # time steps, t_i, i = 0..n-1
        time_stamp = np.arange(0, self._bsde.num_time_interval) * self._bsde.delta_t

        # dw has the same steps as t_i, its first dimension is j, batch size
        # x has one more step than dw and t_i, see Figure 1
        # placeholders need to be fed by feed_dict
        self._dw = tf.placeholder(TF_DTYPE, [None, self._dim, self._num_time_interval], name='dW')
        self._x = tf.placeholder(TF_DTYPE, [None, self._dim, self._num_time_interval + 1], name='X')
        self._is_training = tf.placeholder(tf.bool)

        # y_init is a single value variable, k=1 in Section 3.2 and Section 4.1
        k = 1
        self._y_init = tf.Variable(tf.random_uniform([k],
                                                     minval=self._config.y_init_range[0],
                                                     maxval=self._config.y_init_range[1],
                                                     dtype=TF_DTYPE))

        z_init = tf.Variable(tf.random_uniform([k, self._dim],
                                               minval=-0.1, maxval=0.1,
                                               dtype=TF_DTYPE))

        all_one_vec = tf.ones(shape=tf.stack([tf.shape(self._dw)[0], k]), dtype=TF_DTYPE)  # shape of batch size x k
        y = all_one_vec * self._y_init  # Y(t=0), shape of batch size x k (k=1)
        z = tf.matmul(all_one_vec, z_init)  # Z(t=0), shape of batch size x d

        with tf.variable_scope('forward'):
            for i in xrange(0, self._num_time_interval-1):  # 0..N-2, (25) in page 8
                f = self._bsde.f_tf(time_stamp[i], self._x[:, :, i], y, z)  # f(t) = f(t, x(t), y(t), z(t))
                y = y - self._bsde.delta_t * f + rsum(z * self._dw[:, :, i])  # y(t+1) = y(t) - f(t) dt + z(t) dW(t)
                z = self._subnetwork(self._x[:, :, i + 1], str(i + 1)) / self._dim  # z(t+1) = SUB(x(t+1))/d, (15)

            # terminal time, N-1 to N. I am using N and T interchangeably
            f = self._bsde.f_tf(time_stamp[-1], self._x[:, :, -2], y, z)  # f(T-1) = f(T-1, x(T-1), y(T-1), z(T-1))
            y = y - self._bsde.delta_t * f + rsum(z * self._dw[:, :, -1])  # y(T) = y(T-1) - f(T-1) dt + z(T-1) dW(T-1)

            # loss in (26), page 9
            delta = y - self._bsde.g_tf(self._total_time, self._x[:, :, -1])  # loss = |y(T) - g(T, x(T))|^2
            # use linear approximation outside the clipped range
            clipped_delta = 2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2
            self._loss = tf.reduce_mean(tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta), clipped_delta))

        # train operations
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False, dtype=tf.int32)
        learning_rate = tf.train.piecewise_constant(global_step,
                                                    self._config.lr_boundaries,
                                                    self._config.lr_values)
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self._loss, trainable_variables)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        apply_op = optimizer.apply_gradients(zip(grads, trainable_variables),
                                             global_step=global_step, name='train_step')
        all_ops = [apply_op] + self._extra_train_ops
        self._train_ops = tf.group(*all_ops)
        self._t_build = time.time()-start_time

    def _subnetwork(self, x, name):
        """Builds a sub-network that connects X(t) to Z(t) in the feed forward network.
           See Figure 1 in page 12 of [1]
        """
        with tf.variable_scope(name):
            # standardize the path input first
            # the affine  could be redundant, but helps converge faster
            hiddens = self._batch_norm(x, name='path_input_norm')
            for i in xrange(1, len(self._config.num_hiddens)-1):
                hiddens = self._dense_batch_layer(hiddens,
                                                  self._config.num_hiddens[i],
                                                  activation_fn=tf.nn.relu,
                                                  name='layer_{}'.format(i))
            output = self._dense_batch_layer(hiddens,
                                             self._config.num_hiddens[-1],
                                             activation_fn=None,
                                             name='final_layer')
        return output

    def _dense_batch_layer(self, input_, output_size,
                           activation_fn=None,
                           stddev=5.0, name='linear'):
        """Construct one layer of output = fn(bn(w * input))
            called from _subnetwork()
        """
        with tf.variable_scope(name):
            shape = input_.get_shape().as_list()  # shape[1] is _dim / previous output size
            weight = tf.get_variable('Matrix', [shape[1], output_size], TF_DTYPE,
                                     tf.random_normal_initializer(
                                         stddev=stddev/np.sqrt(shape[1]+output_size)))
            hiddens = tf.matmul(input_, weight)
            hiddens_bn = self._batch_norm(hiddens)
        if activation_fn:
            return activation_fn(hiddens_bn)
        else:
            return hiddens_bn

    def _batch_norm(self, x, name='batch_norm'):
        """Batch normalization:
           input: x: hiddens from _dense_batch_layer()
           output: bn(x)
        """

        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]  # -1 removes time dimension
            beta = tf.get_variable('beta', params_shape, TF_DTYPE,
                                   initializer=tf.random_normal_initializer(
                                       0.0, stddev=0.1, dtype=TF_DTYPE))
            gamma = tf.get_variable('gamma', params_shape, TF_DTYPE,
                                    initializer=tf.random_uniform_initializer(
                                        0.1, 0.5, dtype=TF_DTYPE))
            moving_mean = tf.get_variable('moving_mean', params_shape, TF_DTYPE,
                                          initializer=tf.constant_initializer(0.0, TF_DTYPE),
                                          trainable=False)
            moving_variance = tf.get_variable('moving_variance', params_shape, TF_DTYPE,
                                              initializer=tf.constant_initializer(1.0, TF_DTYPE),
                                              trainable=False)
            # These ops will only be preformed when training
            mean, variance = tf.nn.moments(x, [0], name='moments')
            self._extra_train_ops.append(
                moving_averages.assign_moving_average(moving_mean, mean, MOMENTUM))
            self._extra_train_ops.append(
                moving_averages.assign_moving_average(moving_variance, variance, MOMENTUM))
            mean, variance = tf.cond(self._is_training,
                                     lambda: (mean, variance),
                                     lambda: (moving_mean, moving_variance))
            y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, EPSILON)
            y.set_shape(x.get_shape())
            return y
