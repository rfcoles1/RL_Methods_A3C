import os
import gym
import multiprocessing
import time 

print time.ctime()

class Config:
    ### GAME PARAMS ###
    env_name = 'Acrobot-v1' 
    if env_name == 'CartPole-v0' or env_name == 'MountainCar-v0' or env_name == 'Acrobot-v1':
        mode = 'discrete'
    elif env_name == 'MountainCarContinuous-v0' or env_name == 'Pendulum-v0': 
        mode = 'continuous'

    game = gym.make(env_name)
    s_size = len(game.reset()) 
    if mode == 'discrete':
        a_size = game.action_space.n
    else:
        a_size = game.action_space.shape[0]
        a_bounds = [game.action_space.low, game.action_space.high]
        a_range = game.action_space.high - game.action_space.low

    ### NETWORK PARAMS ###
    gamma = .99
    num_hidden = 128
    lr = 1e-4
    entropy_beta = 0.01
    
    max_episode_len = 1000
    buffer_len = 30 

    ### MODEL PARAMS ###
    num_workers = 2 #multiprocessing.cpu_count() 
    checkpoints = 3
    save_freq = 250
    load_model = False
    model_path = './Models/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path = model_path + str(time.ctime())    

    worker_path = './Workers/'
    if not os.path.exists(worker_path):
        os.makedirs(worker_path)
    worker_path = worker_path + str(time.ctime())
    
    ### VIDEO PARAMS ###
    render_freq = 250
    render = False
    if render == True:
        video_path = './Videos/'
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        video_path = video_path + str(time.ctime())
    

