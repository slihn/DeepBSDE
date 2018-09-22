"""
The main file to run BSDE solver to solve parabolic partial differential equations (PDEs).

"""

import json
import logging
import os
import numpy as np
import tensorflow as tf
from config import get_config
from equation import get_equation
from solver import FeedForwardModel

class OptDef:
    problem_name = 'HJB'
    log_dir = './logs'
    num_run = 1

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('problem_name', OptDef.problem_name,
                           """The name of partial differential equation.""")
tf.app.flags.DEFINE_integer('num_run', OptDef.num_run,
                            """The number of experiments to repeatedly run for the same problem.""")
tf.app.flags.DEFINE_string('log_dir', OptDef.log_dir,
                           """Directory where to write event logs and output array.""")


def main():
    problem_name = FLAGS.problem_name
    log_dir = FLAGS.log_dir
    num_run = FLAGS.num_run
    main2(problem_name, log_dir, num_run)


def main2(problem_name=OptDef.problem_name, num_run=OptDef.num_run, log_dir=OptDef.log_dir):
    config = get_config(problem_name)
    bsde = get_equation(problem_name, config.dim, config.total_time, config.num_time_interval)

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    path_prefix = os.path.join(log_dir, problem_name)

    # save config to file
    with open('{}_config.json'.format(path_prefix), 'w') as outfile:
        json.dump(dict((name, getattr(config, name))
                       for name in dir(config) if not name.startswith('__')),
                  outfile, indent=2)
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)-6s %(message)s')

    for idx_run in range(1, num_run+1):
        tf.reset_default_graph()
        with tf.Session() as sess:
            logging.info('Begin to solve %s with run %d' % (problem_name, idx_run))
            model = FeedForwardModel(config, bsde, sess)
            if bsde.y_init:
                logging.info('Y0_true: %.4e' % bsde.y_init)
            model.build()
            training_history = model.train()
            if bsde.y_init:
                logging.info('relative error of Y0: %s',
                             '{:.2%}'.format(
                                 abs(bsde.y_init - training_history[-1, 2])/bsde.y_init))
            # save training history
            np.savetxt('{}_training_history_{}.csv'.format(path_prefix, idx_run),
                       training_history,
                       fmt=['%d', '%.5e', '%.5e', '%d'],
                       delimiter=",",
                       header="step,loss_function,target_value,elapsed_time",
                       comments='')

if __name__ == '__main__':
    # main()

    # main2("HJB")  # Y0 ~ 4.59, loss ~ 2e-2
    # main2("AllenCahn")  # Y0 ~ 5.3e-2, loss ~ 6e-5
    # main2("PricingOption")  # Y0 ~ 2.13e+01, lss ~ 3.3e+1
    # main2("PricingDefaultRisk")  # Y0 ~ 5.7e+01, loss: 2.6e+1

    main2("BurgesType")
    # if sigma=d/sqrt(2), Y0: 5.2e-1 , loss: 1.3e-3, Y0_true = 0.5, 3.9% error
    # if sigma=d, Y0: 4.98e-1 , loss: 2.7e-3, Y0_true = 0.5, 0.32% error


