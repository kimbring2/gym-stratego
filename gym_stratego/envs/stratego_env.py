import numpy as np
import gym
import pygame
from PIL import Image, ImageTk
from math import sin, pi

from gym_stratego.envs.Army import Army, Icons
import gym_stratego.envs.explosion
from gym_stratego.envs.constants import *  # @UnusedWildImport
from gym_stratego.envs.brains import *  # @UnusedWildImport


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def heart_attack_risk(hypertension, heart_attack_proclivity=0.5):
    return heart_attack_proclivity * sigmoid(hypertension - 6)

def heart_attack_occured(state, heart_attack_proclivity=0.5):
    return np.random.uniform(0, 1) < heart_attack_risk(state['hypertension'], heart_attack_proclivity)

def alertness_decay(time_since_slept):
    return sigmoid((time_since_slept - 40) / 10)

def crippling_anxiety(alertness):
    return sigmoid(alertness - 3)

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def ballmer_function(intoxication):
    return sigmoid((0.05 - intoxication) * 50) + 2 * gaussian(intoxication, 0.135, 0.005)

def wakeup(state):
    state['alertness'] = np.random.uniform(0.7, 1.3)
    state['time_since_slept'] = 0

def drink_coffee(state):
    state['alertness'] += np.random.uniform(0, 1)
    state['hypertension'] += np.random.uniform(0, 0.3)

def drink_beer(state):
    state['alertness'] -= np.random.uniform(0, 0.5)
    state['hypertension'] += np.random.uniform(0, 0.3)
    # source https://attorneydwi.com/b-a-c-per-drink/
    state['intoxication'] += np.random.uniform(0.01, 0.03)

decay_rate = 0.97
half_life = decay_rate ** 24

def half_hour_passed(state):
    state['alertness'] -= alertness_decay(state['time_since_slept'])
    state['hypertension'] = decay_rate * state['hypertension']
    state['intoxication'] = decay_rate * state['intoxication']
    state['time_since_slept'] += 1
    state['time_elapsed'] += 1

def productivity(state):
    p = 1
    p *= state['alertness']
    p *= 1 - crippling_anxiety(state['alertness'])
    p *= ballmer_function(state['intoxication'])
    return p

def work(state):
    state['work_done'] += productivity(state)
    half_hour_passed(state)

def do_nothing(state):
    pass

def sleep(state):
    """Have 16 half-hours of healthy sleep"""
    for hh in range(16):
        half_hour_passed(state)
    wakeup(state)

def make_heartpole_obs_space(observations):
    lower_obs_bound = {
        'alertness': - np.inf,
        'hypertension': 0,
        'intoxication': 0,
        'time_since_slept': 0,
        'time_elapsed': 0,
        'work_done': - np.inf
    }
    higher_obs_bound = {
        'alertness': np.inf,
        'hypertension': np.inf,
        'intoxication': np.inf,
        'time_since_slept': np.inf,
        'time_elapsed': np.inf,
        'work_done': np.inf
    }

    low = np.array([lower_obs_bound[o] for o in observations])
    high = np.array([higher_obs_bound[o] for o in observations])
    shape = (len(observations),)
    
    return gym.spaces.Box(low,high,shape)


class StrategoEnv(gym.Env):
    def __init__(self, heart_attack_proclivity=0.5):
        self.actions = [do_nothing, drink_coffee, drink_beer, sleep]
        self.observations = ['alertness', 'hypertension', 'intoxication',
                             'time_since_slept', 'time_elapsed', 'work_done']
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.observation_space = make_heartpole_obs_space(self.observations)
        self.heart_attack_proclivity = heart_attack_proclivity
        self.log = ''

        size = "Normal"
        self.boardWidth = SIZE_DICT[size][0]
        self.pools = SIZE_DICT[size][1]
        self.tilePix = SIZE_DICT[size][2]
        self.armyHeight = min(4, (self.boardWidth - 2) / 2)
        self.boardsize = self.boardWidth * self.tilePix

        grassImage = pygame.image.load("gym_stratego/envs/%s/%s" % (TERRAIN_DIR, LAND_TEXTURE))
        DEFAULT_IMAGE_SIZE = (self.boardWidth * self.tilePix, self.boardWidth * self.tilePix)
        self.grass_image = pygame.transform.scale(grassImage, DEFAULT_IMAGE_SIZE)

        waterImage = pygame.image.load("gym_stratego/envs/%s/%s" % (TERRAIN_DIR, WATER_TEXTURE))
        DEFAULT_IMAGE_SIZE = (self.tilePix, self.tilePix)
        self.water_image = pygame.transform.scale(waterImage, DEFAULT_IMAGE_SIZE)

        #print("self.boardsize: ", self.boardsize)
        #print("self.armyHeight: ", self.armyHeight)

        self.blueArmy = Army("classical", "Blue", self.boardWidth * self.armyHeight)
        self.redArmy = Army("classical", "Red", self.boardWidth * self.armyHeight)

        self.BLACK = (0, 0, 0)
        self.WHITE = (200, 200, 200)

        pygame.init()
        self.SCREEN = pygame.display.set_mode((self.boardsize, self.boardsize))
        CLOCK = pygame.time.Clock()
        self.SCREEN.fill(self.BLACK)

    def observation(self):
        return np.array([self.state[o] for o in self.observations])
        
    def reset(self):
        self.state = {
            'alertness': 0,
            'hypertension': 0,
            'intoxication': 0,
            'time_since_slept': 0,
            'time_elapsed': 0,
            'work_done': 0
        }
        
        wakeup(self.state)

        return self.observation()
        
    def step(self, action):
        if self.state['time_elapsed'] == 0:
            old_score = 0
        else:
            old_score = self.state['work_done'] / self.state['time_elapsed']
        
        # Do selected action
        self.actions[action](self.state)
        self.log += f'Chosen action: {self.actions[action].__name__}\n'
        
        # Do work
        work(self.state)
        
        new_score = self.state['work_done'] / self.state['time_elapsed']
        
        reward = new_score - old_score
        
        if heart_attack_occured(self.state, self.heart_attack_proclivity):
            # We would like to avoid this
            self.log += 'HEART_ATTACK\n'
            reward -= 100

            # A heart attack is like purgatory - painful, but cleansing
            # You can tell I am not a doctor
            self.state['hypertension'] = 0

        self.log += str(self.state) + '\n'

        pygame.display.update()

        return self.observation(), reward, False, {}
    
    def close(self):
        pass
        
    def isPool(self, x, y):
        """Check whether there is a pool at tile (x,y)."""
        # uneven board size + middle row or even board size + middle 2 rows
        if (self.boardWidth % 2 == 1 and y == self.boardWidth / 2) or \
            ((self.boardWidth % 2 == 0) and (y == self.boardWidth / 2 or y == (self.boardWidth / 2) - 1)):
            return sin(2 * pi * (x + .5) / BOARD_WIDTH * (POOLS + 0.5)) < 0

    def render(self, mode=None):
        blockSize = self.armyHeight # Set the size of the grid block
        #for x in range(0, self.boardsize, blockSize):
        #    for y in range(0, self.boardsize, blockSize):
        #        rect = pygame.Rect(x, y, blockSize, blockSize)
        #        pygame.draw.rect(self.SCREEN, self.WHITE, rect, 1)
        # draw water
        self.SCREEN.blit(self.grass_image, (0, 0))

        for x in range(self.boardWidth):
            for y in range(self.boardWidth):
                if self.isPool(x, y):
                    DEFAULT_IMAGE_POSITION = (x * self.tilePix, y * self.tilePix)
                    self.SCREEN.blit(self.water_image, DEFAULT_IMAGE_POSITION)

        pygame.display.update()

        print(self.log)
        self.log = ''