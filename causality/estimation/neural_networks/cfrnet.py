# vim: foldmethod=marker
""" CFRNET """
from collections import defaultdict

from cfrnet.cfr.cfr_net import cfr_net
from cfrnet.cfr.util import simplex_project
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
            configuration=None,
            optimizer="RMSProp",
            num_iterations=3000,
            num_iterations_per_decay=100,
            validation_fraction=0.1,
            seed=None,
            *args, **kwargs):

        self.configuration = configuration
        if self.configuration is None:
            # use default configuration.

            configuration = {
                "rbf_sigma": 0.1, "dim_in": 200, "wass_lambda": 10.0,
                "decay": 0.3, "dropout_in": 1.0, "weight_init": 0.1,
                "use_p_correction": 0, "batch_size": 100, "rep_weight_decay": 0,
                "wass_iterations": 10, "reweight_sample": 1, "p_alpha": 0,
                "varsel": 0, "n_in": 3, "dim_out": 100,
                "dropout_out": 1.0, "n_out": 3, "p_lambda": 0.0001,
                "lrate_decay": 0.97, "lrate": 0.001, "wass_bpt": 1,
                "batch_norm": 0, "nonlin": "elu", "normalization": "divide",
                "imb_fun": "wass", "split_output": 1, "optimizer": "Adam",
                "pred_output_delay": 0, "loss": "l2", "sparse": 0,
                "varsel": 0, "repetitions": 1, "output_delay": 100
            }

        self.treatment_probability = np.mean(treatment_assignment)
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
        self.CFR = cfr_net(
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

        train_step = opt.minimize(self.CFR.tot_loss, global_step=global_step)
        #  }}} Define Model Graph #

        #  Set up feed dicts for losses {{{ #
        # XXX: Change this to use proper data access
        dict_factual = {
            self.CFR.x: train_data.covariates,
            self.CFR.t: np.expand_dims(train_data.treatment_assignment, 1),
            self.CFR.y_: np.expand_dims(train_data.observed_outcomes, 1),
            self.CFR.do_in: 1.0, self.CFR.do_out: 1.0,
            self.CFR.r_alpha: configuration.p_alpha,
            self.CFR.r_lambda: configuration.p_lambda,
            self.CFR.p_t: self.treatment_probability
        }

        dict_valid = {
            self.CFR.x: validation_data.covariates,
            self.CFR.t: np.expand_dims(validation_data.treatment_assignment, 1),
            self.CFR.y_: np.expand_dims(validation_data.observed_outcomes, 1),
            self.CFR.do_in: 1.0, self.CFR.do_out: 1.0,
            self.CFR.r_alpha: configuration.p_alpha, self.CFR.r_lambda: configuration.p_lambda,
            self.CFR.p_t: self.treatment_probability
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
                    self.CFR.x: x_batch, self.CFR.t: t_batch, self.CFR.y_: y_batch,
                    self.CFR.do_in: configuration.dropout_in,
                    self.CFR.do_out: configuration.dropout_out,
                    self.CFR.r_alpha: configuration.p_alpha,
                    self.CFR.r_lambda: configuration.p_lambda,
                    self.CFR.p_t: self.treatment_probability
                }
            )
            #  }}} Gradient Descent step #

            #  Project variable selection weights {{{ #
            if configuration.varsel:
                wip = simplex_project(tensorflow_session.run(self.CFR.weights_in[0]), 1)
                tensorflow_session.run(self.CFR.projection, feed_dict={self.CFR.w_proj: wip})
            #  }}} Project variable selection weights #

            #  Network objective for training and validation {{{ #
            train_obj_loss, _, _ = tensorflow_session.run(
                [self.CFR.tot_loss, self.CFR.pred_loss, self.CFR.imb_dist],
                feed_dict=dict_factual
            )

            valid_obj_loss, _, _ = tensorflow_session.run(
                [self.CFR.tot_loss, self.CFR.pred_loss, self.CFR.imb_dist],
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

    def predict(self, tensorflow_session, covariates):
        try:
            self.CFR
        except AttributeError:
            # TODO: Message
            raise NotFittedError()
        num_units, _ = covariates.shape

        treated_dict = {
            self.CFR.x: covariates,
            self.CFR.t: np.ones((num_units, 1)),
            self.CFR.do_in: 1.0, self.CFR.do_out: 1.0,
            self.CFR.r_alpha: self.configuration.p_alpha,
            self.CFR.r_lambda: self.configuration.p_lambda,
            self.CFR.p_t: self.treatment_probability
        }

        control_dict = {
            self.CFR.x: covariates,
            self.CFR.t: np.zeros((num_units, 1)),
            self.CFR.do_in: 1.0, self.CFR.do_out: 1.0,
            self.CFR.r_alpha: self.configuration.p_alpha,
            self.CFR.r_lambda: self.configuration.p_lambda,
            self.CFR.p_t: self.treatment_probability
        }

        treated_output = np.asarray(tensorflow_session.run([self.CFR.output], feed_dict=treated_dict))
        control_output = np.asarray(tensorflow_session.run([self.CFR.output], feed_dict=control_dict))

        return treated_output - control_output

    def predict_ite(self, tensorflow_session, covariates):
        return self.predict(tensorflow_session=tensorflow_session, covariates=covariates)

    def predict_ate(self, tensorflow_session, covariates):
        return np.mean(self.predict_ite(tensorflow_session=tensorflow_session, covariates=covariates))

    def plot_train_trajectory(self, marker=None, keys=("train_loss", "val_loss"), titles=("Train", "Validation")):
        import seaborn as sns
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 2)

        keys = ("train_loss", "val_loss")

        progressbar = optional_progressbar(zip(axes, keys, titles), desc="Plotting train trajectory")

        for axis, key, title in progressbar:
            history = self.train_history[key]
            epochs = range(len(history))
            # axis.plot(epochs, history)
            sns.lineplot(x=epochs, y=history, ax=axis, marker=marker)
            axis.set_title(title)
            axis.set_xlabel("Iteration")
            axis.set_ylabel("Loss")

        figure.suptitle("{} - Train Trajectory".format(self.__class__.__name__), size=14)

        return (figure, axes)
