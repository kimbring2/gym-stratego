import gym
import time
import cv2
import numpy as np
import random

from gym_stratego.envs import StrategoEnv
import gym_stratego
from gym_stratego.envs.constants import *

env = gym.make("stratego-v0")
print("env.action_space.n: ", env.action_space.n)
 
enemy_ai = True
human_play = True

for episode in range(0, 100):
    observation = env.reset()

    move = None

    step = 0
    while True:
        print("step: ", step)
        print("env.turn: ", env.turn)
        print("env.step_phase: ", env.step_phase)
        
        if human_play == False:
            start = time.time()

            battle_field = observation['battle_field']
            print("battle_field.shape: ", battle_field.shape)

            battle_field = np.reshape(battle_field, (10, 10))
            print(battle_field)

            possible_actions = observation['possible_actions']
            print("possible_actions: ", possible_actions)

            ego_offboard = observation['ego_offboard']
            print("ego_offboard: ", ego_offboard)

            oppo_offboard = observation['oppo_offboard']
            print("oppo_offboard: ", oppo_offboard)

            action = random.choice(possible_actions)
            print("action: ", action)

            observation, reward, done, info = env.step(action)

            end = time.time()
            #print("elapsed time: ", end - start)
            #print("")
        else:
            if env.turn == 'Blue' and env.step_phase == 1:
                if enemy_ai == True:
                    (oldlocation, move) = env.brains[env.turn].findMove()
                    result = env.move_unit(oldlocation[0], oldlocation[1])
                    env.update_screen()
                else:
                    observation, reward, done, info = env.step_render()
            elif env.turn == 'Blue' and env.step_phase == 2:
                if enemy_ai == True:
                    result = env.move_unit(move[0], move[1])
                    env.update_screen()
                else:
                    observation, reward, done, info = env.step_render()
            else:
                observation, reward, done, info = env.step_render()

                battle_field = observation['battle_field']
                battle_field = np.reshape(battle_field, (10, 10))
                #print(battle_field)
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
                #print("movable_positions: ", movable_positions)

                ego_offboard_rank = observation['ego_offboard_rank']
                #print("ego_offboard_rank: ", ego_offboard_rank)

                oppo_offboard_rank = observation['oppo_offboard_rank']
                #print("oppo_offboard_rank: ", oppo_offboard_rank)

        if done:
            print("reward: ", reward)
            print("done: ", done)
            break

        step += 1

        #time.sleep(1.0)
        print("")

env.close()