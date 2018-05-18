import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

class AC_Network():
    def __init__(self,config,scope,trainer):
        self.config = config
        with tf.variable_scope(scope):
                      
            #input layers           
            self.inputs = tf.placeholder(shape=[None, self.config.s_size], dtype = tf.float32)

            fc1 = slim.fully_connected(self.inputs, self.config.num_hidden,
                activation_fn = tf.nn.relu,
                weights_initializer = tf.contrib.layers.xavier_initializer(),
                biases_initializer = tf.zeros_initializer())

            fc2 = slim.fully_connected(self.inputs, self.config.num_hidden,
                activation_fn = tf.nn.relu,
                weights_initializer = tf.contrib.layers.xavier_initializer(),
                biases_initializer = tf.zeros_initializer())

            #output layers for policy and value
            #actor
            if config.mode == 'discrete':
                self.policy = slim.fully_connected(fc2, self.config.a_size,
                    activation_fn=tf.nn.softmax,
                    weights_initializer = tf.contrib.layers.xavier_initializer(),
                    biases_initializer = None)
            elif config.mode == 'continuous':
                self.mu = slim.fully_connected(fc2, self.config.a_size, 
                    activation_fn=tf.nn.softmax,
                    weights_initializer = tf.contrib.layers.xavier_initializer(),
                    biases_initializer = None)
                self.sigma = slim.fully_connected(fc2, self.config.a_size, 
                    activation_fn=tf.nn.softmax,
                    weights_initializer = tf.contrib.layers.xavier_initializer(),
                    biases_initializer = None)
                self.policy_norm_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
          
            #critic
            self.value = slim.fully_connected(fc2,1,
                activation_fn=None,
                weights_initializer  = tf.contrib.layers.xavier_initializer(),
                biases_initializer = None)
            
            if scope != 'global': #allows a worker access to loss function and gradient update functions
                """
                if config.mode == 'discrete':
                    self.actions = tf.placeholder(shape = [None,], dtype = tf.int32)
                elif config.mode == 'continuous':
                    self.actions = tf.placeholder(shape = [None, self.config.a_size], dtype = tf.float32)
                """             
                self.target_v = tf.placeholder(shape = [None], dtype = tf.float32)
                self.td_loss = self.target_v - tf.reshape(self.value,[-1])
                self.value_loss = 0.5*tf.reduce_sum(tf.square(self.td_loss))

                self.advantages = tf.placeholder(shape = [None], dtype = tf.float32)

                if config.mode == 'discrete':   
                    self.actions = tf.placeholder(shape = [None,], dtype = tf.int32)
                    self.actions_onehot = tf.one_hot(self.actions, self.config.a_size, dtype = tf.float32)
                    self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])
                    self.entropy = -tf.reduce_sum(self.policy * tf.log(self.policy))
                    self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
                elif config.mode == 'continuous':
                    self.actions = tf.placeholder(shape = [None, self.config.a_size], dtype = tf.float32)
                    self.log_prob = self.policy_norm_dist.log_prob(self.actions)
                    self.entropy = self.policy_norm_dist.entropy() 
                    self.policy_loss = -tf.reduce_mean(-(self.entropy + self.log_prob*self.td_loss)) 

                self.loss = 0.5*self.value_loss + self.policy_loss - self.entropy * 0.01
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms, = tf.clip_by_global_norm(self.gradients,40.0)

                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))

    #credit github.com/go2sea 
    def choose_action(self):
        if self.config.mode == 'discrete':
            return tf.multinomial(tf.log(self.policy), 1)[0][0]
        elif self.config.mode == 'continuous':
            sample_action = self.policy_norm_dist.sample(1) * self.config.a_range + self.config.a_bounds[0]
            return tf.clip_by_value(tf.squeeze(sample_action, axis=0), self.config.a_bounds[0][0], self.config.a_bounds[1][0])[0]


