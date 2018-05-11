import numpy as np
import scipy.signal
import tensorflow as tf
import tensorflow.contrib.slim as slim
import threading 
import multiprocessing
import os
import gym 

game = gym.make('CartPole-v0')

class AC_Network():
    def __init__(self,s_size,a_size,scope,trainer):
        with tf.variable_scope(scope):
                      
            #input layers
            self.inputs = tf.placeholder(shape=[None,s_size], dtype = tf.float32)

            #hidden layers
            #fc1 = slim.fully_connected(self.inputs, num_hidden,
            #    activation_fn = tf.nn.relu,
            #    weights_initializer = tf.contrib.layers.xavier_initializer(),
            #    biases_initializer=tf.zeros_initializer())


            #lstm - From tutorial, test w/wout       
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(s_size,state_is_tuple=True)
            c_init = np.zeros((1,lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1,lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1,lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1,lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(self.inputs,[0])
            step_size = tf.shape(self.inputs)[:1]
            state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell,rnn_in, initial_state = state_in, sequence_length=step_size, time_major = False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1,s_size])
            

            #output layers for policy and value
            self.policy = slim.fully_connected(rnn_out,a_size,
                activation_fn=tf.nn.softmax,
                weights_initializer = tf.contrib.layers.xavier_initializer(),
                biases_initializer = None)
            self.value = slim.fully_connected(rnn_out,1,
                activation_fn=None,
                weights_initializer  = tf.contrib.layers.xavier_initializer(),
                biases_initializer = None)
            
            if scope != 'global': #allows a worker access to loss function and gradient update functions
                self.actions = tf.placeholder(shape = [None], dtype = tf.int32)
                self.actions_onehot = tf.one_hot(self.actions, a_size, dtype = tf.float32)
                self.target_v = tf.placeholder(shape = [None], dtype = tf.float32)
                self.advantages = tf.placeholder(shape = [None], dtype = tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                self.value_loss = 0.5*tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.entropy = -tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
                self.loss = 0.5*self.value_loss + self.policy_loss - self.entropy * 0.01

                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms, = tf.clip_by_global_norm(self.gradients,40.0)

                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))

def update_target_graph(from_scope,to_scope): #sets workers parameters to be same as the global network
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

"""
def normalized_columns_initializer(std=1.0): #Why this initializer? Initializes weights
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer
"""

def discount(x,gamma): #calculates discounted returns
    return scipy.signal.lfilter([1],[1,-gamma], x[::-1], axis=0)[::-1]

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
        self.local_AC = AC_Network(s_size, a_size, self.name, trainer)
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
            self.local_AC.advantages:advantages,
            self.local_AC.state_in[0]:self.batch_rnn_state[0],
            self.local_AC.state_in[1]:self.batch_rnn_state[1]}
        #generate network statistics
        v_l,p_l,e_l,g_n,v_n,_ = sess.run([self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.apply_grads],
            feed_dict = feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n


    def work(self,max_episode_length, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print "Starting worker " + str(self.number)
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop(): 
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                d = False #done
                
                s = self.env.reset()
                episode_frames.append(s)
                rnn_state = self.local_AC.state_init
                self.batch_rnn_state = rnn_state
                
                while d == False:
                    #take action according to policy network
                    a_dist,v,rnn_state = sess.run([self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                        feed_dict={self.local_AC.inputs:[s],
                        self.local_AC.state_in[0]:rnn_state[0],
                        self.local_AC.state_in[1]:rnn_state[1]})
                    a = np.random.choice(a_dist[0], p = a_dist[0])
                    a = np.argmax(a_dist == a)
                    

                    s1,r,d,_ = self.env.step(a)                    
                    if d == False:
                        episode_frames.append(s1)
                    else: 
                        s1 = s

                    episode_buffer.append([s,a,r,s1,d,v[0,0]])
                    episode_values.append(v[0,0])

                    episode_reward += r
                    s = s1
                    total_steps += 1
                    episode_step_count += 1

                    #if episode hasnt ended, but experience buffer is full
                    #make update using that experience rollout
                    if len(episode_buffer) == 30 and d != True and episode_step_count != max_episode_length -1:
                        #value estimation
                        v1 = sess.run(self.local_AC.value,
                            feed_dict={self.local_AC.inputs:[s],
                            self.local_AC.state_in[0]:rnn_state[0],
                            self.local_AC.state_in[1]:rnn_state[1]})[0,0]
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
                    #summary.value.add(tag='Losses/Value_Loss', simple_value=float(v_l))
                    #summary.value.add(tag='Losses/Policy_Loss', simple_value=float(p_l))
                    #summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    #summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    #summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))

                    self.summary_writer.add_summary(summary,episode_count)
                    self.summary_writer.flush()

                #why
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1
                

                     
gamma = .99
num_hidden = 100
lr = 1e-4
checkpoints = 5
#needs to generalized for any game
max_episode_length = 300

s_size = len(game.reset())
a_size = game.action_space.n


load_model = False
model_path = './model'
if not os.path.exists(model_path):
    os.makedirs(model_path)


tf.reset_default_graph()

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0,dtype=tf.int32, name = 'global_episodes', trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=lr)
    master_network = AC_Network(s_size,a_size,'global',None)

    num_workers = multiprocessing.cpu_count()
    workers = []
    for i in range(num_workers):
        workers.append(Worker(gym.make('CartPole-v0'),i,s_size,a_size,trainer,model_path,global_episodes))
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
        t = threading.Thread(target = worker.work(max_episode_length, gamma, sess, coord,saver))
        t.start()
        worker_threads.append(t)
    coord.join(worker_threads)
