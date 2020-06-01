from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def build_network(state_dim):
    state_input = Input(shape=(state_dim,), dtype='float64')
    h1 = Dense(64, activation='relu')(state_input)
    h2 = Dense(32, activation='relu')(h1)
    h3 = Dense(16, activation='relu')(h2)
    v_output = Dense(1, activation='linear')(h3)
    model = Model(state_input, v_output)
    return model, model.trainable_weights, state_input

class GlobalCritic(object):
    def __init__(self, state_dim, learning_rate):
        self.state_dim = state_dim
        self.model, self.phi, _ = build_network(state_dim)

        self.optimizer = Adam(learning_rate)

    # save Actor parameter
    def save_weights(self, path):
        self.model.save_weights(path)

    # load Actor parameter
    def load_weights(self, path):
        self.model.load_weights(path+'pendulum_critic.h5')


class WorkerCritic(object):
    def __init__(self, state_dim, action_dim, global_critic):

        self.global_critic = global_critic

        self.state_dim = state_dim
        self. action_dim = action_dim

        # build critic neural net
        self.model, self.phi, self.states = build_network(self.state_dim)

    def train(self, states, td_targets):
        with tf.GradientTape() as g:
            # loss
            v_values = self.model(states)
            loss = tf.reduce_sum(tf.square(td_targets - v_values))

        # gradient
        dj_dphi = g.gradient(loss, self.phi)

        # clip gradient
        dj_dphi, _ = tf.clip_by_global_norm(dj_dphi, 40)
        grads = zip(dj_dphi, self.global_critic.phi)
        self.global_critic.optimizer.apply_gradients(grads)

    # save Critic parameter
    def save_weights(self, path):
        self.model.save_weights(path)

    # load Critic parameter
    def load_weights(self, path):
        self.model.load_weights(path + 'pendulum_critic.h5')