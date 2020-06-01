import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

"""
    tf 2.0 with the "functional API"
"""

# Actor neural network
def build_network(state_dim, action_dim, action_bound):
    state_input = Input(shape=(state_dim,), dtype='float64')
    h1 = Dense(64, activation='relu')(state_input)
    h2 = Dense(32, activation='relu')(h1)
    h3 = Dense(16, activation='relu')(h2)
    out_mu = Dense(action_dim, activation='tanh')(h3)
    std_output = Dense(action_dim, activation='softplus')(h3)

    # bound mean
    mu_output = Lambda(lambda x: x*action_bound)(out_mu)
    model = Model(state_input, [mu_output, std_output])
    return model, model.trainable_weights, state_input

class GlobalActor(object):

    def __init__(self, state_dim, action_dim, action_bound, learning_rate):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound

        # build global actor neural net
        self.model, self.theta, self.states = build_network(self.state_dim, self.action_dim, self.action_bound)

        #optimizer
        self.optimizer = Adam(learning_rate)

        # calculate mean
    def predict(self, state):
        mu_a, _ = self.model(np.reshape(state, [1, self.state_dim]))
        return mu_a[0]

    # save Actor parameter
    def save_weights(self, path):
        self.model.save_weights(path)

    # load Actor parameter
    def load_weights(self, path):
        self.model.load_weights(path+'pendulum_actor.h5')


class WorkerActor(object):

    def __init__(self, state_dim, action_dim, action_bound, entropy_beta, global_actor):
        self.global_actor = global_actor

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.entropy_beta = entropy_beta

        # set min and max of standard deviation
        self.std_bound = [1e-2, 1.0]

        # create actor neural net
        self.model, self.theta, self.states = build_network(self.state_dim, self.action_dim, self.action_bound)


    # log_policy pdf
    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / var - 0.5 * tf.math.log(var * 2 * np.pi)
        entropy = 0.5 * (tf.math.log(2 * np.pi * std ** 2) + 1.0)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True), tf.reduce_sum(entropy, 1, keepdims=True)

    def train(self, states, actions, advantages):
        with tf.GradientTape() as g:
            mu_a, std_a = self.model(states)
            log_policy_pdf, entropy = self.log_pdf(mu_a, std_a, actions)
            # loss
            loss_policy = log_policy_pdf * advantages
            loss = tf.reduce_sum(-loss_policy - self.entropy_beta * entropy)
        dj_dtheta = g.gradient(loss, self.theta)

        # clip gradient
        dj_dtheta, _ = tf.clip_by_global_norm(dj_dtheta, 40)

        # train global neural network
        grads = zip(dj_dtheta, self.global_actor.theta)
        self.global_actor.optimizer.apply_gradients(grads)

    # get action
    def get_action(self, state):
        mu_a, std_a = self.model(np.reshape(state, [1, self.state_dim]))
        mu_a = mu_a[0]
        std_a = std_a[0]

        std_a = np.clip(std_a, self.std_bound[0], self.std_bound[1])
        action = np.random.normal(mu_a, std_a, size=self.action_dim)
        return action
