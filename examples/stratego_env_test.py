import gym
import time
import cv2
import numpy as np
import random

from gym_stratego.envs import StrategoEnv
import gym_stratego

env = gym.make("stratego-v0")
#print("env.observation_space: ", env.observation_space)
#print("env.action_space: ", env.action_space)

human_play = True

for episode in range(0, 100):
    observation = env.reset()

    step = 0
    while True:
        print("step: ", step)
        
        if human_play == False:
            battle_field = observation['battle_field']
            print("battle_field.shape: ", battle_field.shape)

            battle_field = np.reshape(battle_field, (10, 10))
            print(battle_field)

            current_turn = observation["current_turn"]
            print("current_turn: ", current_turn)

            possible_actions = observation['possible_actions']
            print("possible_actions: ", possible_actions)

            red_offboard = observation['red_offboard']
            print("red_offboard: ", red_offboard)

            blue_offboard = observation['blue_offboard']
            print("blue_offboard: ", blue_offboard)

            action = random.choice(possible_actions)

            observation, reward, done, info = env.step(action)

        else:
            observation, reward, done, info = env.step_render()

            battle_field = observation['battle_field']
            print("battle_field.shape: ", battle_field.shape)

            blue_offboard = observation['blue_offboard']
            print("blue_offboard: ", blue_offboard)

            movable_units = observation['movable_units']
            print("movable_units: ", movable_units)

            clicked_unit = observation['clicked_unit']
            print("clicked_unit: ", clicked_unit)

            movable_positions = observation['movable_positions']
            print("movable_positions: ", movable_positions)

            red_offboard_rank = observation['red_offboard_rank']
            print("red_offboard_rank: ", red_offboard_rank)

            blue_offboard_rank = observation['blue_offboard_rank']
            print("blue_offboard_rank: ", blue_offboard_rank)


        if done:
            print("done: ", done)
            break

        step += 1

env.close()