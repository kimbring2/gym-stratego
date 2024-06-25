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
            start = time.time()

            battle_field = observation['battle_field']
            #print("battle_field.shape: ", battle_field.shape)

            red_battle_field = np.reshape(battle_field, (10, 10))
            #print(battle_field)

            current_turn = observation["current_turn"]
            #print("current_turn: ", current_turn)

            possible_actions = observation['possible_actions']
            #print("possible_actions: ", possible_actions)

            red_offboard = observation['red_offboard']
            #print("red_offboard: ", red_offboard)

            blue_offboard = observation['blue_offboard']
            #print("blue_offboard: ", blue_offboard)

            action = random.choice(possible_actions)

            observation, reward, done, info = env.step(action)

            end = time.time()
            print("elapsed time: ", end - start)
            print("")
            #time.sleep(0.5)
        else:
            observation, reward, done, info = env.step_render()

            battle_field = observation['battle_field']
            battle_field = np.reshape(battle_field, (10, 10))
            print(battle_field)
            #print("battle_field.shape: ", battle_field.shape)

            blue_offboard = observation['blue_offboard']
            #print("blue_offboard: ", blue_offboard)

            movable_units = observation['movable_units']
            #print("movable_units: ", movable_units)

            clicked_unit = observation['clicked_unit']
            #print("clicked_unit: ", clicked_unit)

            movable_positions = observation['movable_positions']
            #print("movable_positions: ", movable_positions)

            red_offboard_rank = observation['red_offboard_rank']
            #print("red_offboard_rank: ", red_offboard_rank)

            blue_offboard_rank = observation['blue_offboard_rank']
            #print("blue_offboard_rank: ", blue_offboard_rank)

        if done:
            print("reward: ", reward)
            print("done: ", done)
            break

        step += 1

env.close()