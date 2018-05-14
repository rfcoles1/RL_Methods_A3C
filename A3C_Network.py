import numpy as np
import scipy.signal
import tensorflow as tf
import tensorflow.contrib.slim as slim
import threading 
import multiprocessing
import os
import gym 

game = gym.make('CartPole-v0')

num_hidden = 100

class AC_Network():
    def __init__(self,s_size,a_size,scope,trainer):
        with tf.variable_scope(scope):
                      
            #input layers           
            self.inputs = tf.placeholder(shape=[None,s_size], dtype = tf.float32)

            fc1 = slim.fully_connected(self.inputs, num_hidden,
                activation_fn = tf.nn.relu,
                weights_initializer = tf.contrib.layers.xavier_initializer(),
                biases_initializer = tf.zeros_initializer())

            fc2 = slim.fully_connected(self.inputs, num_hidden,
                activation_fn = tf.nn.relu,
                weights_initializer = tf.contrib.layers.xavier_initializer(),
                biases_initializer = tf.zeros_initializer())

            #output layers for policy and value
            self.policy = slim.fully_connected(fc2,a_size,
                activation_fn=tf.nn.softmax,
                weights_initializer = tf.contrib.layers.xavier_initializer(),
                biases_initializer = None)
            self.value = slim.fully_connected(fc2,1,
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

    def get_policy(self,state):
        return self.sess.run(self.policy, {self.inputs:state})

    def get_value(self,state):
        return self.sess.run(self.value, {self.inputs:state})

    def get_policy_value(self,sess,state):
        policy, value = sess.run([self.policy, self.value], {self.inputs:state})
        return policy, value
 
