#from gym_stratego.GymStratego import GymStratego

from gym.envs.registration import register 

register(id='stratego-v0', 
	     entry_point='gym_stratego.envs:StrategoEnv') 