import gym
import time
import cv2
import numpy as np
import random
import argparse

from gym_stratego.envs import StrategoEnv
import gym_stratego
from gym_stratego.envs.constants import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--enemy_ai', type=bool, default=False, help='AI Enemy')
parser.add_argument('--human_play', type=bool, default=True, help='Play as human')
arguments = parser.parse_args()

env = gym.make("stratego-v0")
 
enemy_ai = arguments.enemy_ai
human_play = arguments.human_play


for episode in range(0, 100):
    print("episode: ", episode)

    observation = env.reset()

    done = False
    move = None

    step = 0
    while True:
        env.update_screen()
            
        if human_play == False:
            if env.turn == 'Blue':
                if enemy_ai == True:
                    (oldlocation, move) = env.brains[env.turn].findMove()
                    result = env.move_unit(oldlocation[0], oldlocation[1])
                    result = env.move_unit(move[0], move[1])

                    observation, _, _, info = env.small_observation()
                    done = env.done
                    reward = env.reward
                    if done:
                        break
                else:
                    battle_field = observation['battle_field']
                    battle_field = np.reshape(battle_field, (10, 10))
                    possible_actions = observation['possible_actions']
                    ego_offboard = observation['ego_offboard']
                    oppo_offboard = observation['oppo_offboard']
                    action = random.choice(possible_actions)
                    observation, _, _, info = env.step(action)
            else:
                battle_field = observation['battle_field']
                battle_field = np.reshape(battle_field, (10, 10))

                battle_field = np.reshape(battle_field, (10, 10))

                possible_actions = observation['possible_actions']

                ego_offboard = observation['ego_offboard']

                oppo_offboard = observation['oppo_offboard']

                action = random.choice(possible_actions)
                observation, _, _, info = env.step(action)
                done = env.done
                reward = env.reward
                if done:
                    break
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

                ego_offboard = observation['ego_offboard']
                oppo_offboard = observation['oppo_offboard']
                movable_units = observation['movable_units']
                clicked_unit = observation['clicked_unit']
                movable_positions = observation['movable_positions']
                ego_offboard_rank = observation['ego_offboard_rank']
                oppo_offboard_rank = observation['oppo_offboard_rank']

        step += 1

        #time.sleep(1.0)

env.close()