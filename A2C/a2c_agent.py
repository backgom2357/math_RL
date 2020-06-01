import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from A2C.a2c_actor import Actor
from A2C.a2c_critic import Critic

class A2Cagent(object):

    def __init__(self, env):

        # hyperparameter
        self.GAMMA = 0.95
        self.BATCH_SIZE = 32
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001

        # environment
        self.env = env
        # state dimension
        self.state_dim = env.observation_space.shape[0]
        # action dimension
        self.action_dim = env.action_space.shape[0]
        # max action size
        self.action_bound = env.action_space.high[0]

        # create Actor and Critic neural nets
        self.actor = Actor(self.state_dim, self.action_dim, self.action_bound, self.ACTOR_LEARNING_RATE)
        self.critic = Critic(self.state_dim, self.action_dim, self.CRITIC_LEARNING_RATE)

        # total reward of a episode
        self.save_epi_reward = []

    # calculate advantages and TD targets
    def advantage_td_target(self, reward, v_value, next_v_value, done):
        if done:
            y_k = reward
            advantage = y_k - v_value
        else:
            y_k = reward + self.GAMMA * next_v_value
            advantage = y_k - v_value
        return advantage, y_k

    # extract data from batch
    def unpack_batch(self, batch):
        unpack = batch[0]
        for idx in range(len(batch)-1):
            unpack = np.append(unpack, batch[idx+1], axis=0)
        return unpack

    # train agent
    def train(self, max_episode_num):

        # repeat for each episode
        for ep in range(int(max_episode_num)):

            # init batch
            batch_state, batch_action, batch_td_target, batch_advantage = [], [], [] ,[]
            # init episode
            time, episode_reward, done = 0, 0, False
            # reset env and observe initial state
            state = self.env.reset()

            while not done:

                # visualize env
                # self.env.render()

                # get action
                action = self.actor.get_action(state)

                # bound action range
                action = np.clip(action, -self.action_bound, self.action_bound)

                # observe next state, reward
                next_state, reward, done, _ = self.env.step(action)

                # reshape
                state = np.reshape(state, [1, self.state_dim])
                next_state = np.reshape(next_state, [1, self.state_dim])
                action = np.reshape(action, [1, self.action_dim])
                reward = np.reshape(reward, [1, 1])

                # calculate state value
                v_value = self.critic.model(state)
                next_v_value = self.critic.model(next_state)

                # calculate advantage and TD target
                train_reward = (reward+8)/8
                advantage, y_i = self.advantage_td_target(train_reward, v_value, next_v_value, done)

                # append to batch
                batch_state.append(state)
                batch_action.append(action)
                batch_td_target.append(y_i)
                batch_advantage.append(advantage)

                # wait for full batch
                if len(batch_state) < self.BATCH_SIZE:

                    # update state
                    state = next_state[0]
                    episode_reward += reward[0]
                    time += 1
                    continue

                # train
                # extract from batch
                states = self.unpack_batch(batch_state)
                actions = self.unpack_batch(batch_action)
                td_targets = self.unpack_batch(batch_td_target)
                advantages = self.unpack_batch(batch_advantage)

                # clear batch
                batch_state, batch_action, batch_td_target, batch_advantage = [], [], [], []

                # critic neural net update
                self.critic.train_on_batch(states, td_targets)

                # actor neural net update
                self.actor.train(states, actions, advantages)

                # update state
                state = next_state[0]
                episode_reward += reward[0]
                time += 1

            print('Episode: ', ep+1, 'Time: ', time, 'Reward: ', episode_reward)
            self.save_epi_reward.append(episode_reward)

            # save neural net parameters
            if ep % 10 == 0:
                self.actor.save_weights("./save_weights/pendulum_actor.h5")
                self.critic.save_weights("./save_weights/pendulum_critic.h5")
                state_for_check = np.reshape(state, [1, self.state_dim])
                print("Critic value: ", self.critic.model(state_for_check))

        # save total reward
        np.savetxt("./save_weights/pendulum_epi_reward.txt", self.save_epi_reward)

    def test(self):

        # Initialize model

        # init batch
        batch_state, batch_action, batch_td_target, batch_advantage = [], [], [], []
        # init episode
        time, episode_reward, done = 0, 0, False
        # reset env and observe initial state
        state = self.env.reset()

        while not done:

            # visualize env
            # self.env.render()

            # get action
            action = self.actor.get_action(state)

            # bound action range
            action = np.clip(action, -self.action_bound, self.action_bound)

            # observe next state, reward
            next_state, reward, done, _ = self.env.step(action)

            # reshape
            state = np.reshape(state, [1, self.state_dim])
            next_state = np.reshape(next_state, [1, self.state_dim])
            action = np.reshape(action, [1, self.action_dim])
            reward = np.reshape(reward, [1, 1])

            # calculate state value
            v_value = self.critic.model(state)
            next_v_value = self.critic.model(next_state)

            # calculate advantage and TD target
            train_reward = (reward + 8) / 8
            advantage, y_i = self.advantage_td_target(train_reward, v_value, next_v_value, done)

            # append to batch
            batch_state.append(state)
            batch_action.append(action)
            batch_td_target.append(y_i)
            batch_advantage.append(advantage)

            # wait for full batch
            if len(batch_state) < self.BATCH_SIZE:
                # update state
                state = next_state[0]
                episode_reward += reward[0]
                time += 1
                continue

            # train
            # extract from batch
            states = self.unpack_batch(batch_state)
            actions = self.unpack_batch(batch_action)
            td_targets = self.unpack_batch(batch_td_target)
            advantages = self.unpack_batch(batch_advantage)

            # clear batch
            batch_state, batch_action, batch_td_target, batch_advantage = [], [], [], []

            # critic neural net update
            self.critic.train_on_batch(states, td_targets)

            # actor neural net update
            self.actor.train(states, actions, advantages)

            # update state
            state = next_state[0]
            episode_reward += reward[0]
            time += 1

        self.actor.load_weights('./save_weights/')
        self.critic.load_weights('./save_weights/')


    # graph episodes and rewards
    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()


















