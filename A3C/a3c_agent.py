import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import threading
import multiprocessing

from a3c_actor import GlobalActor, WorkerActor
from a3c_critic import GlobalCritic, WorkerCritic

# global variables
global_episode_count = 0 # total episode count
global_step = 0 # total step count
global_episode_reward = [] # save result


class A3Cagent(object):
    """
        build global neural net
    """

    def __init__(self, env_name):
        # environment
        self.env_name = env_name
        self.WORKERS_NUM = multiprocessing.cpu_count() # set the number of workers

        # hyperparameters
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001

        env = gym.make(self.env_name)

        # state dimension
        state_dim = env.observation_space.shape[0]

        # action dimension
        action_dim = env.action_space.shape[0]

        # action bound
        action_bound = env.action_space.high[0]

        # build global actor and critic neural net
        self.global_actor = GlobalActor(state_dim, action_dim, action_bound, self.ACTOR_LEARNING_RATE)
        self.global_critic = GlobalCritic(state_dim, self.CRITIC_LEARNING_RATE)

    def train(self, max_episode_num):
        workers = []

        # create worker thread and add to list
        for i in range(self.WORKERS_NUM):
            worker_name = 'worker%i' % i
            workers.append(A3Cworker(worker_name, self.env_name, self.global_actor, self.global_critic, max_episode_num))

        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()

        np.savetxt('./save_weights/pendulum_epi_reward.txt', global_episode_reward)
        print(global_episode_reward)

    def plot_result(self):
        plt.plot(global_episode_reward)
        plt.show()


class A3Cworker(threading.Thread):
    """
        build worker thread
    """
    def __init__(self, worker_name, env_name, global_actor, global_critic, max_episode_num):
        threading.Thread.__init__(self)

        # hyperparameters
        self.GAMMA = 0.95
        self.ENTROPY_BETA = 0.01
        self.t_MAX = 4 # n-step

        self.max_episode_num = max_episode_num

        # create env of worker
        self.env = gym.make(env_name)
        self.worker_name = worker_name

        # share global neural net
        self.global_actor = global_actor
        self.global_critic = global_critic

        # state dimension
        self.state_dim = self.env.observation_space.shape[0]

        # action dimension
        self.action_dim = self.env.action_space.shape[0]

        # action bound
        self.action_bound = self.env.action_space.high[0]

        # build actor and critic neural net
        self.worker_actor = WorkerActor(self.state_dim, self.action_dim, self.action_bound, self.ENTROPY_BETA, self.global_actor)
        self.worker_critic = WorkerCritic(self.state_dim, self.action_dim, self.global_critic)

        # copy global neural net parameters and apply to worker neural network
        self.worker_actor.model.set_weights(self.global_actor.model.get_weights())
        self.worker_critic.model.set_weights(self.global_critic.model.get_weights())

    # calculate n-step TD target
    def n_step_td_target(self, rewards, next_v_value, done):
        td_targets = np.zeros_like(rewards)
        cumulative = 0
        if not done:
            cumulative = next_v_value
        for k in reversed(range(0, len(rewards))):
            cumulative = self.GAMMA * cumulative + rewards[k]
            td_targets = cumulative
        return td_targets

    # extract data from batch
    def unpack_batch(self, batch):
        unpack = batch[0]
        for idx in range(len(batch) - 1):
            unpack = np.append(unpack, batch[idx + 1], axis=0)
        return unpack

    def run(self):

        global global_step, global_episode_count, global_episode_reward

        # print when worker run
        print(self.worker_name, "----------------------------starts")

        # repeat per episode
        while global_episode_count <= int(self.max_episode_num):

            # init states, actions, rewards
            states, actions, rewards = [], [], []
            # init episode
            step, episode_reward, done = 0, 0, False
            # init env and state
            state = self.env.reset()

            # episode
            while not done:

                # env render
                # self.env.render()
                # get action
                action = self.worker_actor.get_action(state)
                # clip action bound
                action = np.clip(action, -self.action_bound, self.action_bound)
                # observe next state and reward
                next_state, reward, done, _ = self.env.step(action)
                # save to batch
                states.append(state)
                actions.append(action)
                rewards.append((reward+8)/8) # modify reward range
                # state update
                state = next_state
                step += 1
                episode_reward += reward

                if len(states) == self.t_MAX or done:

                    # get data from batch
                    states = np.array(states)
                    actions = np.array(actions)
                    rewards = np.array(rewards)

                    # calculate n-step TD target and advantage
                    next_state = np.reshape(next_state, [1, self.state_dim])
                    next_v_value = self.worker_critic.model(next_state)
                    n_step_td_targets = self.n_step_td_target(rewards, next_v_value, done)
                    v_values = self.worker_critic.model(states)
                    advantages = n_step_td_targets - v_values

                    # global critic and actor neural net update
                    self.worker_critic.train(states, n_step_td_targets)
                    self.worker_actor.train(states, actions, advantages)

                    # apply global neural net parameter to worker neural net
                    self.worker_actor.model.set_weights(self.global_actor.model.get_weights())
                    self.worker_critic.model.set_weights(self.global_critic.model.get_weights())

                    # global step update
                    global_step += 1

                    # clear batch
                    states, actions, rewards = [], [], []

                # episode end
                if done:
                    # update global episode count
                    global_episode_count += 1

                    # print("Worker name: {}, Epi: {}, Step: {}, Reward: {}".format(self.worker_name, global_episode_count, step, episode_reward))
                    print("Worker name: ", self.worker_name, "Epi: ", global_episode_count, "Step: ", step, "Reward: ", episode_reward)
                    global_episode_reward.append(episode_reward)

                    # save global neural net parameter
                    if global_episode_count % 10 == 0:
                        chck_state = np.reshape(state, [1, self.state_dim])
                        print("Actor value: ", self.global_actor.model(chck_state), "Critic value: ", self.global_critic.model(chck_state))
                        self.global_actor.save_weights("./save_weights/pendulum_actor.h5")
                        self.global_critic.save_weights("./save_weights/pendulum_critic.h5")










