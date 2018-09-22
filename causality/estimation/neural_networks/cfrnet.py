# vim: foldmethod=marker
""" CFRNET """
from collections import defaultdict

from cfr.cfr_net import cfr_net
from cfr.util import simplex_project
import numpy as np
import tensorflow as tf

from causality.data.datasets.dataset import Dataset
from causality.exceptions import NotFittedError
from causality.estimation.estimator import Estimator
from causality.utils import optional_progressbar


class CFRNet(Estimator):
    @property
    def train_history(self):
        try:
            return self.history
        except AttributeError:
            # TODO: Add message
            raise NotFittedError()

    def fit(self,
            tensorflow_session,
            covariates,
            observed_outcomes,
            treatment_assignment,
            configuration,
            optimizer="RMSProp",
            num_iterations=3000,
            num_iterations_per_decay=100,
            validation_fraction=0.1,
            seed=None,
            *args, **kwargs):

        treatment_probability = np.mean(treatment_assignment)
        _, num_covariates = covariates.shape

        dataset = Dataset(
            covariates=covariates,
            treatment_assignment=treatment_assignment,
            observed_outcomes=observed_outcomes
        )

        train_data, validation_data = dataset.split(
            validation_fraction=validation_fraction, seed=configuration.seed
        )

        #  Set up placeholders {{{ #

        placeholders = {
            "covariates": tf.placeholder(
                "float", shape=[None, num_covariates], name='covariates'
            ),
            "treatment_assignment": tf.placeholder(
                "float", shape=[None, 1], name='treatment_assignment'
            ),
            "responses": tf.placeholder(
                "float", shape=[None, 1], name='responses'
            )
        }
        #  }}} Set up placeholders #

        #  CFRNet Parameters {{{ #
        r_alpha = tf.placeholder("float", name='r_alpha')
        r_lambda = tf.placeholder("float", name='r_lambda')
        do_in = tf.placeholder("float", name='dropout_in')
        do_out = tf.placeholder("float", name='dropout_out')
        p = tf.placeholder("float", name='p_treated')

        #  }}} CFRNet Parameters #

        #  Define Model Graph {{{ #
        dimensions = [num_covariates, configuration.dim_in, configuration.dim_out]
        CFR = cfr_net(
            x=placeholders["covariates"],
            t=placeholders["treatment_assignment"],
            y_=placeholders["responses"],
            p_t=p,
            FLAGS=configuration,
            r_alpha=r_alpha,
            r_lambda=r_lambda,
            do_in=do_in,
            do_out=do_out,
            dims=dimensions,
        )

        ''' Set up optimizer '''
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(
            configuration.lrate, global_step,
            num_iterations_per_decay, configuration.lrate_decay, staircase=True
        )

        if configuration.optimizer == 'Adagrad':
            opt = tf.train.AdagradOptimizer(lr)
        elif configuration.optimizer == 'GradientDescent':
            opt = tf.train.GradientDescentOptimizer(lr)
        elif configuration.optimizer == 'Adam':
            opt = tf.train.AdamOptimizer(lr)
        else:
            opt = tf.train.RMSPropOptimizer(lr, configuration.decay)

        train_step = opt.minimize(CFR.tot_loss, global_step=global_step)
        #  }}} Define Model Graph #

        #  Set up feed dicts for losses {{{ #
        # XXX: Change this to use proper data access
        dict_factual = {
            CFR.x: train_data.covariates,
            CFR.t: np.expand_dims(train_data.treatment_assignment, 1),
            CFR.y_: np.expand_dims(train_data.observed_outcomes, 1),
            CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.r_alpha: configuration.p_alpha,
            CFR.r_lambda: configuration.p_lambda, CFR.p_t: treatment_probability
        }

        dict_valid = {
            CFR.x: validation_data.covariates,
            CFR.t: np.expand_dims(validation_data.treatment_assignment, 1),
            CFR.y_: np.expand_dims(validation_data.observed_outcomes, 1),
            CFR.do_in: 1.0, CFR.do_out: 1.0,
            CFR.r_alpha: configuration.p_alpha, CFR.r_lambda: configuration.p_lambda,
            CFR.p_t: treatment_probability
        }

        tensorflow_session.run(tf.global_variables_initializer())

        #  }}} Set up feed dicts for losses #

        self.history = defaultdict(list)

        batch_generator = train_data.generate_batches(
            batch_size=configuration.batch_size
        )

        progressbar = optional_progressbar(
            range(num_iterations), total=num_iterations, desc="Iterations"
        )
        for iteration in progressbar:
            batch_data = next(batch_generator)
            x_batch = batch_data.covariates
            t_batch = np.expand_dims(batch_data.treatment_assignment, 1)
            y_batch = np.expand_dims(batch_data.observed_outcomes, 1)

            #  Gradient Descent step {{{ #
            tensorflow_session.run(
                train_step, feed_dict={
                    CFR.x: x_batch, CFR.t: t_batch, CFR.y_: y_batch,
                    CFR.do_in: configuration.dropout_in,
                    CFR.do_out: configuration.dropout_out,
                    CFR.r_alpha: configuration.p_alpha,
                    CFR.r_lambda: configuration.p_lambda,
                    CFR.p_t: treatment_probability
                }
            )
            #  }}} Gradient Descent step #

            #  Project variable selection weights {{{ #
            if configuration.varsel:
                wip = simplex_project(tensorflow_session.run(CFR.weights_in[0]), 1)
                tensorflow_session.run(CFR.projection, feed_dict={CFR.w_proj: wip})
            #  }}} Project variable selection weights #

            #  Network objective for training and validation {{{ #
            train_obj_loss, _, _ = tensorflow_session.run(
                [CFR.tot_loss, CFR.pred_loss, CFR.imb_dist],
                feed_dict=dict_factual
            )

            valid_obj_loss, _, _ = tensorflow_session.run(
                [CFR.tot_loss, CFR.pred_loss, CFR.imb_dist],
                feed_dict=dict_valid
            )

            self.history["train_loss"].append(train_obj_loss)
            self.history["val_loss"].append(valid_obj_loss)

            if iteration % 100 == 0:
                progressbar.set_postfix_str(
                    "losses - Train: {} Validation: {}".format(
                        train_obj_loss, valid_obj_loss
                    )
                )

            #  }}} Network objective train, validation #

        return self

    def predict(self, covariates):
        pass
