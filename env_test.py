import gym
import time
import cv2
import numpy as np

env = gym.make("gym_stratego.envs:stratego-v0")

while True:
    observation = env.reset()
    battle_field = observation['battle_field']  / 255.0

    for step in range(100000000):
        #cv2.imshow('battle_field', battle_field)
        #cv2.waitKey(1)

        print("step: ", step)

        #env.render()

        action = 0
        #print("action: ", action)

        observation, reward, done, step_phase = env.step(action)

        battle_field = observation['battle_field'] / 255.0
        red_offboard = observation['red_offboard']
        blue_offboard = observation['blue_offboard']
        movable_units = observation['movable_units']

        print("battle_field.shape: ", battle_field.shape)
        print("red_offboard: ", red_offboard)
        print("blue_offboard: ", blue_offboard)
        print("movable_units: ", movable_units)

        print("step_phase: ", step_phase)
        print("reward: ", reward)
        print("done: ", done)

        if done:
            break

        #time.sleep(1)
        print("")

env.close()