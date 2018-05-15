import numpy as np
import scipy.signal
import tensorflow as tf
import tensorflow.contrib.slim as slim
import threading 
import multiprocessing
import os
import gym 

from A3C_Network import AC_Network

env_name = 'MountainCar-v0'
game = gym.make(env_name)


def update_target_graph(from_scope,to_scope): #sets workers parameters to be same as the global network
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


def discount(rewards,gamma): #calculates discounted returns
    return scipy.signal.lfilter([1],[1,-gamma],rewards[::-1],axis=0)[::-1]


class Worker():
    def __init__(self, game, name, s_size, a_size, trainer, model_path, global_episodes):
        self.name = 'worker_' + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_" + str(self.number))

        #initalizes game
        self.env = game

        #gives local copy of network
        self.local_AC = AC_Network(s_size, a_size, num_hidden, self.name, trainer)
        self.update_local_ops = update_target_graph('global',self.name)

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:,0]
        actions = rollout[:,1]
        rewards = rollout[:,2]
        next_observations = rollout[:,3]
        values = rollout[:,5]

        #generate advantage and discounted returns 
        #uses generalized advantage esitamtor 
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma*self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages,gamma)

        
        #update global network using gradients from loss
        feed_dict = {self.local_AC.target_v:discounted_rewards,
            self.local_AC.inputs:np.vstack(observations),
            self.local_AC.actions:actions,
            self.local_AC.advantages:advantages}
        #generate network statistics
        v_l,p_l,e_l,g_n,v_n,_ = sess.run([self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.apply_grads],
            feed_dict = feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n


    def work(self,max_episode_len, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print "Starting worker " + str(self.number)
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop(): 
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = np.empty(max_episode_len)
                episode_frames = np.empty([max_episode_len, s_size])
                episode_reward = 0
                episode_step_count = 0
                d = False #done
                
                s = self.env.reset()
                episode_frames[0] = s
                
                for i in range(max_episode_len):
                    #take action according to policy network
                    policy,value = sess.run([self.local_AC.policy, self.local_AC.value],
                        feed_dict={self.local_AC.inputs:[s]})
                    #self.local_AC.get_policy_value(sess,s)
                    a = np.random.choice(policy[0], p = policy[0])
                    a = np.argmax(policy == a)
                    

                    s1,r,d,_ = self.env.step(a)                    
                    if d == False:
                        episode_frames[i+1] = s1
                    else: 
                        s1 = s

                    episode_buffer.append([s,a,r,s1,d,value[0,0]])
                    episode_values[i] = value[0,0]

                    episode_reward += r
                    s = s1
                    total_steps += 1
                    episode_step_count += 1

                    #if episode hasnt ended, but experience buffer is full
                    #make update using that experience rollout
                    if len(episode_buffer) == buffer_len and d != True and episode_step_count != max_episode_len -1:
                        #value estimation
                        v1 = sess.run(self.local_AC.value,
                            feed_dict={self.local_AC.inputs:[s]})
                        v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer, sess, gamma, v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                
                    if d == True:
                        break
                
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                #update network at end of epidsode
                if len(episode_buffer) != 0:
                    v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer, sess, gamma, 0.0)

                #save stats
                if episode_count % 250 == 0 and episode_count != 0:
                    saver.save(sess,self.model_path+'/model-'+str(episode_count)+'cptk')
                    print "saved model"
    
                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
#add summary class?
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value_Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy_Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))

                    self.summary_writer.add_summary(summary,episode_count)
                    self.summary_writer.flush()

                #why

                #if self.name == 'worker_0':
                 #   sess.run(self.increment)
                episode_count += 1
                

                     
gamma = .99
num_hidden = 128
lr = 1e-4
checkpoints = 5
#needs to generalized for any game
max_episode_len = 300
buffer_len = 30

s_size = len(game.reset())
#the following allows the network to work with both discrete and continuous actions spaces
actions = game.action_space
if not actions: #if
    a_size = actions.shape[0]
else:
    a_size = actions.n 


load_model = False
model_path = './model'
if not os.path.exists(model_path):
    os.makedirs(model_path)


tf.reset_default_graph()

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0,dtype=tf.int32, name = 'global_episodes', trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=lr)
    master_network = AC_Network(s_size,a_size,num_hidden,'global',None)

    num_workers = multiprocessing.cpu_count()
    workers = []
    for i in range(num_workers):
        workers.append(Worker(gym.make(env_name),i,s_size,a_size,trainer,model_path,global_episodes))
    saver = tf.train.Saver(max_to_keep=checkpoints)
        

with tf.Session() as sess:
    coord = tf.train.Coordinator()

    if load_model == True:
        print 'Loading Model'
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    worker_threads = []
    for worker in workers: 
        worker_work = lambda: worker.work(max_episode_len, gamma, sess,coord,saver)
        t = threading.Thread(target = (worker_work))
        t.start()
        worker_threads.append(t)
    coord.join(worker_threads)

