import gym
import time

env = gym.make("gym_stratego.envs:stratego-v0")
env.action_space.seed(42)

observation = env.reset()


while True:
    observation = env.reset()
    for step in range(100000000):
        print("step: ", step)

        env.render()

        action = env.action_space.sample()
        #print("action: ", action)

        observation, reward, done, step_phase = env.step(env.action_space.sample())
        print("step_phase: ", step_phase)
        print("reward: ", reward)
        print("done: ", done)

        if done:
            break

        #time.sleep(0.1)
        print("")

env.close()