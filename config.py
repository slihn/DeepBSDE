import numpy as np


# Config class specifies the NN parameters
# main reference:
#    [1] https://arxiv.org/abs/1706.04702
#    [2] https://arxiv.org/abs/1707.02568
class Config(object):
    n_layer = 4  # see page 10-11 of [1]
    batch_size = 64  # this is j in page 10 of [1]
    valid_size = 256  # why is this much bigger?

    step_boundaries = [2000, 4000]
    num_iterations = 6000
    logging_frequency = 100
    verbose = True
    y_init_range = [0, 1]  # TODO maybe this belongs to Equation class?

    # to be defined by children
    # https://www.tensorflow.org/api_docs/python/tf/train/piecewise_constant
    lr_values = None  # learning rate values
    lr_boundaries = None  # learning rate boundaries
    num_hiddens = None


class AllenCahnConfig(Config):
    # according to Section 4.2 of [1]
    total_time = 0.3  # this is T in [1]
    num_time_interval = 20  # this is n in [1]
    dim = 100  # this is d in [1]
    lr_values = list(np.array([5e-4, 5e-4]))  # this is gamma_m in [1]
    lr_boundaries = [2000]
    num_iterations = 6000  # this is m in [1]
    num_hiddens = [dim, dim + 10, dim + 10, dim]  # see page 10 of [1]
    y_init_range = [0.3, 0.6]


class HJBConfig(Config):
    # according to Section 4.3 of [1]
    # Y_0 is about 4.5901.
    dim = 100
    total_time = 1.0
    num_time_interval = 20  # increases from 20 to 100 for better precision?
    lr_values = list(np.array([1e-2, 1e-2]))
    lr_boundaries = [400]
    num_iterations = 4000
    num_hiddens = [dim, dim+10, dim+10, dim]
    y_init_range = [0, 1]


class PricingOptionConfig(Config):
    # according to Section 4.4 of [1]
    dim = 100
    total_time = 0.5
    num_time_interval = 20
    lr_values = list(np.array([5e-3, 5e-3]))
    lr_boundaries = [2000]
    num_iterations = 4000
    num_hiddens = [dim, dim+10, dim+10, dim]
    y_init_range = [15, 18]


class PricingDefaultRiskConfig(Config):
    dim = 100
    total_time = 1
    num_time_interval = 40
    lr_values = list(np.array([8e-3, 8e-3]))
    lr_boundaries = [3000]
    num_iterations = 6000
    num_hiddens = [dim, dim+10, dim+10, dim]
    y_init_range = [40, 50]


class BurgesTypeConfig(Config):
    dim = 50  # (60), page 20 of [1]
    total_time = 0.2
    num_time_interval = 30
    lr_values = list(np.array([1e-2, 1e-3, 1e-4]))
    lr_boundaries = [15000, 25000]
    num_iterations = 30000  # this takes a long time, 2000 seconds
    num_hiddens = [dim, dim+10, dim+10, dim]
    y_init_range = [2, 4]


class QuadraticGradientsConfig(Config):
    dim = 100
    total_time = 1.0
    num_time_interval = 30
    lr_values = list(np.array([5e-3, 5e-3]))
    lr_boundaries = [2000]
    num_iterations = 4000
    num_hiddens = [dim, dim+10, dim+10, dim]
    y_init_range = [2, 4]


class ReactionDiffusionConfig(Config):
    dim = 100
    total_time = 1.0
    num_time_interval = 30
    lr_values = list(np.array([1e-2, 1e-2, 1e-2]))
    lr_boundaries = [8000, 16000]
    num_iterations = 24000
    num_hiddens = [dim, dim+10, dim+10, dim]


def get_config(name):
    try:
        return globals()[name+'Config']
    except KeyError:
        raise KeyError("Config for the required problem not found.")
