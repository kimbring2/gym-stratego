import gym
import time

env = gym.make("gym_stratego.envs:stratego-v0")
env.action_space.seed(42)

observation = env.reset()

for _ in range(100000000):
    env.render()

    action = env.action_space.sample()
    #print("action: ", action)

    observation, reward, done, _ = env.step(env.action_space.sample())
    #print("observation: ", observation)

    if done:
        observation = env.reset()

    time.sleep(0.1)
    #print("")

env.close()