import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

class AC_Network():
    def __init__(self,config,scope,trainer):
        self.config = config
        with tf.variable_scope(scope):
            print self.config.a_size 
            #input layers           
            self.inputs = tf.placeholder(shape=[None, self.config.s_size], dtype = tf.float32)

            fc1 = slim.fully_connected(self.inputs, self.config.num_hidden,
                activation_fn = tf.nn.relu,
                weights_initializer = tf.contrib.layers.xavier_initializer(),
                biases_initializer = tf.zeros_initializer())

            fc2 = slim.fully_connected(fc1, self.config.num_hidden,
                activation_fn = tf.nn.relu,
                weights_initializer = tf.contrib.layers.xavier_initializer(),
                biases_initializer = tf.zeros_initializer())

            #output layers for policy and value
            #actor
            if self.config.mode == 'discrete':
                self.policy = slim.fully_connected(fc2, self.config.a_size,
                    activation_fn=tf.nn.softmax,
                    weights_initializer = tf.contrib.layers.xavier_initializer(),
                    biases_initializer = None)
            elif self.config.mode == 'continuous':
                self.mu = slim.fully_connected(fc2, self.config.a_size, 
                    activation_fn=tf.nn.tanh,
                    weights_initializer = tf.contrib.layers.xavier_initializer(),
                    biases_initializer = None)
                self.sigma = slim.fully_connected(fc2, self.config.a_size, 
                    activation_fn=tf.nn.softplus,
                    weights_initializer = tf.contrib.layers.xavier_initializer(),
                    biases_initializer = None)
                self.policy_norm_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
          
            #critic
            self.value = slim.fully_connected(fc2,1,
                activation_fn=None,
                weights_initializer  = tf.contrib.layers.xavier_initializer(),
                biases_initializer = None)
            
            if scope != 'global': #allows a worker access to loss function and gradient update functions
                self.target_v = tf.placeholder(shape = [None], dtype = tf.float32)
                self.td_loss = tf.reshape(self.value,[-1]) - self.target_v
                self.value_loss = 0.5*tf.reduce_sum(tf.square(self.td_loss))

                self.advantages = tf.placeholder(shape = [None], dtype = tf.float32)

                if config.mode == 'discrete':   
                    self.actions = tf.placeholder(shape = [None,], dtype = tf.int32)
                    self.actions_onehot = tf.one_hot(self.actions, self.config.a_size, dtype = tf.float32)
                    self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])
                    self.entropy = -tf.reduce_sum(self.policy * tf.log(self.policy))#shannon entropy
                    self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
                    self.A = tf.multinomial(tf.log(self.policy), 1)[0][0]

                elif config.mode == 'continuous':
                    self.actions = tf.placeholder(shape = [None, self.config.a_size], dtype = tf.float32)
                    self.log_prob = self.policy_norm_dist.log_prob(self.actions)
                    self.entropy = tf.reduce_sum(self.policy_norm_dist.entropy())
                    self.policy_loss = -tf.reduce_mean(-(self.entropy + self.log_prob*self.td_loss))
                    self.A = tf.clip_by_value(tf.squeeze(self.policy_norm_dist.sample(1), axis=0), self.config.a_bounds[0][0], self.config.a_bounds[1][0])

                self.loss = 0.5*self.value_loss + self.policy_loss - self.entropy * self.config.entropy_beta
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.gradients,_ = tf.clip_by_global_norm(self.gradients, 40.0)
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(self.gradients,global_vars))

                    
