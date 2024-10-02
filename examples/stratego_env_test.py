import gym
import time
import cv2
import numpy as np
import random

from gym_stratego.envs import StrategoEnv
import gym_stratego

env = gym.make("stratego-v0")
print("env.action_space.n: ", env.action_space.n)

human_play = False

for episode in range(0, 100):
    observation = env.reset()

    step = 0
    while True:
        print("step: ", step)
        
        if human_play == False:
            #time.sleep(1.0)
            start = time.time()

            battle_field = observation['battle_field']
            print("battle_field.shape: ", battle_field.shape)

            battle_field = np.reshape(battle_field, (10, 10))
            print(battle_field)

            possible_actions = observation['possible_actions']
            print("possible_actions: ", possible_actions)

            ego_offboard = observation['ego_offboard']
            #print("red_offboard: ", red_offboard)

            oppo_offboard = observation['oppo_offboard']
            #print("blue_offboard: ", blue_offboard)

            action = random.choice(possible_actions)
            print("action: ", action)

            observation, reward, done, info = env.step(action)

            end = time.time()
            #print("elapsed time: ", end - start)
            #print("")
        else:
            observation, reward, done, info = env.step_render()
            #print(info)

            battle_field = observation['battle_field']
            battle_field = np.reshape(battle_field, (10, 10))
            print(battle_field)
            #print("battle_field.shape: ", battle_field.shape)

            ego_offboard = observation['ego_offboard']
            #print("ego_offboard: ", ego_offboard)

            oppo_offboard = observation['oppo_offboard']
            #print("oppo_offboard: ", oppo_offboard)

            movable_units = observation['movable_units']
            #print("movable_units: ", movable_units)

            clicked_unit = observation['clicked_unit']
            #print("clicked_unit: ", clicked_unit)

            movable_positions = observation['movable_positions']
            print("movable_positions: ", movable_positions)

            ego_offboard_rank = observation['ego_offboard_rank']
            #print("ego_offboard_rank: ", ego_offboard_rank)

            oppo_offboard_rank = observation['oppo_offboard_rank']
            #print("oppo_offboard_rank: ", oppo_offboard_rank)

        if done:
            print("reward: ", reward)
            print("done: ", done)
            break

        step += 1

env.close()