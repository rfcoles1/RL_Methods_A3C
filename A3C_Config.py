import os
import gym
import multiprocessing

class Config:
    env_name = 'Acrobot-v1'
    if env_name == 'CartPole-v0' or env_name == 'MountainCar-v0' or env_name == 'Acrobot-v1':
        mode = 'discrete'
    elif env_name == 'MountainCarContinuous-v0' or env_name == 'Pendulum-v0': 
        mode = 'continuous'
    #discrete/continuous

    gamma = .99
    num_hidden = 128
    lr = 1e-4
    entropy_beta = 0.01
    checkpoints = 3

    save_freq = 100
    max_episode_len = 300
    buffer_len = 30

    load_model = False
    model_path = './model'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    num_workers = 2#multiprocessing.cpu_count() 

    game = gym.make(env_name)
    s_size = len(game.reset()) 
    if mode == 'discrete':
        a_size = game.action_space.n
    else:
        a_size = game.action_space.shape[0]
        a_bounds = [game.action_space.low, game.action_space.high]
        a_range = game.action_space.high - game.action_space.low
