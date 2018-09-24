import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal


TF_DTYPE = tf.float64
xrange = range  # python 2.0 migrated to 3.0


# main reference:
#    [1] https://arxiv.org/abs/1706.04702
#    [2] https://arxiv.org/abs/1707.02568
class Equation(object):
    """Base class for defining PDE related function."""

    def __init__(self, dim, total_time, num_time_interval):
        self._dim = dim
        self._total_time = total_time
        self._num_time_interval = num_time_interval
        self._delta_t = (self._total_time + 0.0) / self._num_time_interval
        self._sqrt_delta_t = np.sqrt(self._delta_t)

        self._y_init = None
        self._x_init = np.zeros(self._dim)
        self._sigma = np.sqrt(2.0)
        self._lambda = 1.0

    def sample(self, batch_size):
        """Sample forward SDE."""
        raise NotImplementedError

    def f_tf(self, t, x, y, z):
        """Generator function in the PDE."""
        raise NotImplementedError

    def g_tf(self, t, x):
        """Terminal condition of the PDE."""
        raise NotImplementedError

    @property
    def y_init(self):
        return self._y_init

    @property
    def dim(self):
        return self._dim

    @property
    def num_time_interval(self):
        return self._num_time_interval

    @property
    def total_time(self):
        return self._total_time

    @property
    def delta_t(self):
        return self._delta_t

    def dw_sample_normal(self, batch_size):
        dw_sample = normal.rvs(size=[batch_size,
                                     self._dim,
                                     self._num_time_interval]) * self._sqrt_delta_t
        return dw_sample

    def x_sample_bm(self, batch_size, dw_sample):
        # X(t) = vectorize(self._x_init)
        # X(t+1) = X(t) + sigma * dW(t), see Section 5.4 of [1]
        x_sample = np.zeros([batch_size, self._dim, self._num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([batch_size, self._dim]) * self._x_init
        for i in xrange(self._num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self._sigma * dw_sample[:, :, i]
        return x_sample

    def sample_bm_normal(self, batch_size):
        dw_sample = self.dw_sample_normal(batch_size)
        x_sample = self.x_sample_bm(batch_size, dw_sample)
        return dw_sample, x_sample


def get_equation(name, dim, total_time, num_time_interval):
    try:
        return globals()[name](dim, total_time, num_time_interval)
    except KeyError:
        raise KeyError("Equation for the required problem not found.")


def rsum(x):
    return tf.reduce_sum(x, 1, keepdims=True)


class AllenCahn(Equation):
    def __init__(self, dim, total_time, num_time_interval):
        super(AllenCahn, self).__init__(dim, total_time, num_time_interval)

    def sample(self, batch_size):
        return self.sample_bm_normal(batch_size)

    def f_tf(self, t, x, y, z):
        return y - tf.pow(y, 3)  # Section 4.2, page 12 of [1]

    def g_tf(self, t, x):
        return 0.5 / (1 + 0.2 * rsum(tf.square(x)))  # Section 4.2, page 12 of [1]


class HJB(Equation):
    def __init__(self, dim, total_time, num_time_interval):
        super(HJB, self).__init__(dim, total_time, num_time_interval)
        self._y_init = 4.5901

    def sample(self, batch_size):
        return self.sample_bm_normal(batch_size)

    def f_tf(self, t, x, y, z):
        return -self._lambda * rsum(tf.square(z))  # Section 4.3, page 13 of [1]

    def g_tf(self, t, x):
        return tf.log((1 + rsum(tf.square(x))) / 2)  # Section 4.3, page 13 of [1]


class PricingOption(Equation):
    def __init__(self, dim, total_time, num_time_interval):
        super(PricingOption, self).__init__(dim, total_time, num_time_interval)
        self._x_init = np.ones(self._dim) * 100
        self._sigma = 0.2
        self._mu_bar = 0.06
        self._rl = 0.04  # R_l: lending rate
        self._rb = 0.06  # R_b: borrow rate
        self._alpha = 1.0 / self._dim

    def sample(self, batch_size):
        dw_sample = self.dw_sample_normal(batch_size)

        # (52) of [1], gbm, X(t+1) = e^(sigma dW(t) + drift) * X(t)
        # drift = (mu - sigma^2/2) dt
        x_sample = np.zeros([batch_size, self._dim, self._num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([batch_size, self._dim]) * self._x_init
        drift = (self._mu_bar-(self._sigma**2)/2) * self._delta_t
        for i in xrange(self._num_time_interval):
            x_sample[:, :, i + 1] = np.exp(self._sigma * dw_sample[:, :, i] + drift) * x_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):  # (53) in [1]
        z_sum = rsum(z) / self._sigma
        y_max = tf.maximum(z_sum - y, 0)
        return -self._rl * y - (self._mu_bar - self._rl) * z_sum + (self._rb - self._rl) * y_max

    def g_tf(self, t, x):  # (51) in [1]
        x_max = tf.reduce_max(x, 1, keepdims=True)
        return tf.maximum(x_max - 120, 0) - 2 * tf.maximum(x_max - 150, 0)


class PricingDefaultRisk(Equation):
    # page 5 of [2]
    def __init__(self, dim, total_time, num_time_interval):
        super(PricingDefaultRisk, self).__init__(dim, total_time, num_time_interval)
        self._x_init = np.ones(self._dim) * 100.0
        self._sigma = 0.2
        self._rate = 0.02   # interest rate R
        self._delta = 2.0 / 3
        self._gammah = 0.2
        self._gammal = 0.02
        self._mu_bar = 0.02
        self._vh = 50.0
        self._vl = 70.0
        self._slope = (self._gammah - self._gammal) / (self._vh - self._vl)

    def sample(self, batch_size):
        dw_sample = self.dw_sample_normal(batch_size)

        # TODO: the following needs a reference?
        # it seems to be X(t+1) = (1+ mu*dt + sigma dW(t)) X(t)
        x_sample = np.zeros([batch_size, self._dim, self._num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([batch_size, self._dim]) * self._x_init
        for i in xrange(self._num_time_interval):
            x_sample[:, :, i + 1] = (1 + self._mu_bar * self._delta_t) * x_sample[:, :, i] + (
                self._sigma * x_sample[:, :, i] * dw_sample[:, :, i])
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):  # (9) and (10) of [2]
        piecewise_linear = tf.nn.relu(
            tf.nn.relu(y - self._vh) * self._slope + self._gammah - self._gammal) + self._gammal
        return (-(1 - self._delta) * piecewise_linear - self._rate) * y

    def g_tf(self, t, x):
        return tf.reduce_min(x, 1, keepdims=True)


class BurgesType(Equation):
    def __init__(self, dim, total_time, num_time_interval):
        super(BurgesType, self).__init__(dim, total_time, num_time_interval)
        self._y_init = 1 - 1.0 / (1 + np.exp(0 + np.sum(self._x_init) / self._dim))  # (61), page 21 of [1]
        self._sigma = self._dim * 1.0 # this is different from what's in page 20 of [1]
        # self._sigma = self._dim / np.sqrt(2.0) # changed from d to d/sqrt(2) to match what's in page 20 of [1]

    def sample(self, batch_size):
        return self.sample_bm_normal(batch_size)

    def f_tf(self, t, x, y, z):  # (57) in Section 4.3 of [1]
        return (y - (2 + self._dim) / 2.0 / self._dim) * rsum(z)

    def g_tf(self, t, x):  # (58) in Section 4.3 of [1], and 1-1/(1+X) = X/(1+X)
        return 1 - 1.0 / (1 + tf.exp(t + rsum(x) / self._dim))


class QuadraticGradients(Equation):
    # Section 4.6 of [1]
    def __init__(self, dim, total_time, num_time_interval):
        super(QuadraticGradients, self).__init__(dim, total_time, num_time_interval)
        self._alpha = 0.4
        self._sigma = 1.0  # added by slihn
        self._x_init = np.zeros(self._dim)
        base = self._total_time + np.sum(np.square(self._x_init) / self._dim)
        self._y_init = np.sin(np.power(base, self._alpha))  # this is just sin(1) = 0.84147

    def sample(self, batch_size):
        return self.sample_bm_normal(batch_size)

    def f_tf(self, t, x, y, z):  # (75), Section 4.6 in [1]
        x_square = rsum(tf.square(x))
        base = self._total_time - t + x_square / self._dim
        base_alpha = tf.pow(base, self._alpha)
        derivative = self._alpha * tf.pow(base, self._alpha - 1) * tf.cos(base_alpha)

        # TODO the following needs a reference
        term1 = rsum(tf.square(z))
        term2 = -4.0 * (derivative ** 2) * x_square / (self._dim ** 2)
        term3 = derivative
        term4 = -0.5 * (
            2.0 * derivative + 4.0 / (self._dim ** 2) * x_square * self._alpha * (
                (self._alpha - 1) * tf.pow(base, self._alpha - 2) * tf.cos(base_alpha) - (
                    self._alpha * tf.pow(base, 2 * self._alpha - 2) * tf.sin(base_alpha)
                    )
                )
            )
        return term1 + term2 + term3 + term4

    def g_tf(self, t, x):  # g(x) above (75), Section 4.6 in [1]
        return tf.sin(tf.pow(rsum(tf.square(x)) / self._dim, self._alpha))


class ReactionDiffusion(Equation):
    # Section 4.7 of [1]
    def __init__(self, dim, total_time, num_time_interval):
        super(ReactionDiffusion, self).__init__(dim, total_time, num_time_interval)
        self._sigma = 1.0  # added by slihn
        self._kappa = 0.6
        self._lambda = 1 / np.sqrt(self._dim)
        self._x_init = np.zeros(self._dim)

        exp_term = np.exp(-(self._lambda ** 2) * self._dim * self._total_time / 2)
        sin_term = np.sin(self._lambda * np.sum(self._x_init))
        self._y_init = 1 + self._kappa + sin_term * exp_term  # from (79)

    def sample(self, batch_size):
        return self.sample_bm_normal(batch_size)

    def f_tf(self, t, x, y, z):  # (77) on page 25 of [1]
        exp_term = tf.exp((self._lambda ** 2) * self._dim * (t - self._total_time) / 2)
        sin_term = tf.sin(self._lambda * rsum(x))
        f_tf2 = tf.square(y - self._kappa - 1 - sin_term * exp_term)
        return tf.minimum(tf.constant(1.0, dtype=TF_DTYPE), f_tf2)

    def g_tf(self, t, x):  # page 25 of [1]
        return 1 + self._kappa + tf.sin(self._lambda * rsum(x))

