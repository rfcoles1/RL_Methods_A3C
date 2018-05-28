import numpy as np
import scipy.signal
import tensorflow as tf
import tensorflow.contrib.slim as slim
import threading 
import os
import gym 
from A3C_Config import Config
from A3C_Network import AC_Network

import moviepy.editor as mpy

"""
credit 
github.com/awjuliani for A3C framework and discrete network
github.com/go2sea for discrete/continuous distinction
"""

#helper functions
#sets workers parameters to be same as the global network
def update_target_graph(from_scope,to_scope): 
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

#calculates discounted returns
def discount(rewards,gamma): 
    return scipy.signal.lfilter([1],[1,-gamma],rewards[::-1],axis=0)[::-1]

class Worker():
    def __init__(self, name, config, trainer, global_episodes):
        self.name = 'worker_' + str(name)
        self.number = name
        self.config = config
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter(self.config.worker_path + "train_" + str(self.number))

        self.env = gym.make(self.config.env_name)

        #gives local copy of network
        self.local_AC = AC_Network(self.config, self.name, trainer)
        self.update_local_ops = update_target_graph('global',self.name)

    def train(self, batch, sess, bootstrap_value):
        batch = np.array(batch)
        observations = batch[:,0]
        actions = batch[:,1]
        rewards = batch[:,2]
        next_observations = batch[:,3] 
        #batch[:,4] not needed
        values = batch[:,5]

        #generate advantage and discounted returns 
        #uses generalized advantage estimator 
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,self.config.gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + self.config.gamma*self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages,self.config.gamma)

        if self.config.mode == 'continuous':
            actions = np.array(np.vstack(actions))

        #update global network using gradients from loss
        feed_dict = {self.local_AC.target_v:discounted_rewards.flatten(),
            self.local_AC.inputs:np.vstack(observations),
            self.local_AC.actions:actions,
            self.local_AC.advantages:advantages}
        #generate network statistics
        v_l,p_l,e_l,_ = sess.run([self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.apply_grads],
            feed_dict = feed_dict)

        return v_l/len(batch), p_l/len(batch), e_l/len(batch)


    def work(self,sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print "Starting worker " + str(self.number)
        
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop(): 
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = np.empty(self.config.max_episode_len)
                episode_frames = np.empty([self.config.max_episode_len, self.config.s_size])
                episode_reward = 0
                episode_step_count = 0
                done = False 
                
                s = self.env.reset()
                episode_frames[0] = s

                for i in range(self.config.max_episode_len):
                    a,value = sess.run([self.local_AC.A, self.local_AC.value],
                        feed_dict={self.local_AC.inputs:[s]})
                    s1,r,done,_ = self.env.step(a)    
                    s1 = s1.flatten()

                    episode_frames[i] = s1 
                    episode_buffer.append([s,a,r,s1,done,value[0,0]])
                    episode_values[i] = value[0,0]

                    episode_reward += r
                    s = s1
                    total_steps += 1
                    episode_step_count += 1

                    #if episode hasnt ended, but experience buffer is full
                    #make update using that experience rollout
                    if len(episode_buffer) == self.config.buffer_len and done != True and episode_step_count != self.config.max_episode_len -1:
                        #value estimation
                        v1 = sess.run(self.local_AC.value,
                            feed_dict={self.local_AC.inputs:[s]})
                        v_l,p_l,e_l = self.train(episode_buffer, sess, v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if done == True:
                        break
                
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                #update network at end of episode
                if len(episode_buffer) != 0:
                    v_l,p_l,e_l = self.train(episode_buffer, sess, 0.0)

                #save stats
                if episode_count % self.config.save_freq == 0 and episode_count != 0:                   
                    saver.save(sess, self.config.model_path + '/model-' + str(episode_count) + 'cptk')
                    print "saved model"
                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value_Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy_Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))

                    self.summary_writer.add_summary(summary, episode_count)
                    self.summary_writer.flush()
                    
                    frames = np.array(episode_frames)
                    np.savetxt(self.config.video_path + str(episode_count) + '.dat', frames)

                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1



tf.reset_default_graph()
config = Config()
    
with tf.device("/cpu:0"):   
    global_episodes = tf.Variable(0, dtype=tf.int32, name = 'global_episodes', trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=config.lr)
    master_network = AC_Network(config, 'global', None)

    workers = []
    for i in range(config.num_workers):
        workers.append(Worker(i, config, trainer, global_episodes))
    saver = tf.train.Saver(max_to_keep=config.checkpoints)
        
with tf.Session() as sess:
    coord = tf.train.Coordinator()

    if config.load_model == True:
        ckpt = tf.train.get_checkpoint_state(config.model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
   
    worker_threads = []
    for worker in workers: 
        worker_work = lambda: worker.work(sess,coord,saver)
        t = threading.Thread(target = worker_work)
        t.start()
        worker_threads.append(t)
    coord.join(worker_threads)

