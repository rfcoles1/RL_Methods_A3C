import os
import sys

import gym
sys.path.insert(0,'../..')
import Games

class Config:
    env_name = 'MountainCar-v0'
    
    game = gym.make(env_name)
    s_size = game.observation_space.shape[0]
    a_size = game.action_space.n

    gamma = 0.9
    lr = 1e-4
    update_global_freq = 25

    max_ep = 5000
