from a3c_agent import A3Cagent
import tensorflow as tf

def main():

    tf.keras.backend.set_floatx('float64')

    max_episode_num = 1000
    env_name = "Pendulum-v0"
    agent = A3Cagent(env_name)

    # train
    agent.train(max_episode_num)

    # show plot
    agent.plot_result()

if __name__ == "__main__":
    main()