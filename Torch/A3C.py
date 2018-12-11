import os
import sys
import numpy as np

import torch.multiprocessing as mp

from A3C_Config import Config
from A3C_Network import Network
from A3C_Helper import *
from shared_adam import SharedAdam

import gym

config = Config()

os.environ["OMP_NUM_THREADS"] = "1"

class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Network()           # local network
        self.env = gym.make(config.env_name).unwrapped

    def run(self):
        total_step = 1
        while self.g_ep.value < config.max_ep:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            while True:
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s1, r, done, _ = self.env.step(a)
                if done: 
                    r = -1
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)
                
                if total_step % config.update_global_freq == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s1, buffer_s, buffer_a, buffer_r)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s1
                total_step += 1
        self.res_queue.put(None)


if __name__ == "__main__":
    gnet = Network()        # global network
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters())      # global optimizer
    
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()


    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    #workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, 0)]
  
    [w.start() for w in workers]
    
    results = []
    while True:
        r = res_queue.get()
        if r is not None:
            results.append(r)
        else:
            break
    
    [w.join() for w in workers]

    import matplotlib.pyplot as plt
    plt.plot(results)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()
