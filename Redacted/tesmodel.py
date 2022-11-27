import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import tensorflow_probability as tfp


class Actor(keras.Model):
    def __init__(self, action_space):
        super(Actor, self).__init__()

        self.actor = keras.Sequential()
        # self.actor.add(layers.Dense(512, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
        self.actor.add(
            layers.Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
        self.actor.add(
            layers.Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
        self.actor.add(layers.Dense(action_space, activation="tanh"))

        # OUTPUT
        self.log_std = tf.Variable(shape=(1, action_space), trainable=True, initial_value=tf.zeros(shape=(1, action_space)))

    def __call__(self, observation):
        _input = np.array(observation, dtype=np.float32)

        actor_dense = self.actor(_input)

        mean = actor_dense
        log_std = tf.multiply(self.log_std, 1.0)
        std = tf.exp(log_std)
        dist = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=std)

        return dist

class Critic(keras.Model):
    def __init__(self, input_shape):
        super(Critic, self).__init__()

        self.value = keras.Sequential()
        # self.value.add(
        #     layers.Dense(512, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
        self.value.add(
            layers.Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
        self.value.add(
            layers.Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
        self.value.add(layers.Dense(1, activation=None))

    def __call__(self, observation):
        _input = np.array(observation, dtype=np.float32)

        value_output = self.value(_input)

        return value_output

class Wonder(keras.Model):
    def __init__(self, state_shape):
        super(Wonder, self).__init__()

        self.wonder = keras.Sequential()
        self.wonder.add(layers.Dense(512, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
        self.wonder.add(
            layers.Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
        self.wonder.add(
            layers.Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
        self.wonder.add(layers.Dense(state_shape, activation="tanh"))

    def __call__(self, observation):
        state_input = np.array(observation[0], dtype=np.float32)
        action_input = np.array(observation[1], dtype=np.float32)

        input_merged = layers.concatenate((state_input, action_input))

        wonder = self.wonder(input_merged)

        return wonder

class ActorW(keras.Model):
    def __init__(self, action_space, input_shape):
        super(ActorW, self).__init__()

        self.actor = keras.Sequential()
        self.actor.add(layers.Dense(512, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
        self.actor.add(
            layers.Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
        self.actor.add(
            layers.Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
        self.actor.add(layers.Dense(action_space, activation="tanh"))

        # OUTPUT
        self.log_std = tf.Variable(shape=(1, action_space), trainable=True, initial_value=tf.zeros(shape=(1, action_space)))

        # self.steps = keras.Sequential()
        # self.steps.add(
        #     layers.Dense(512, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
        # self.steps.add(
        #     layers.Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
        # self.steps.add(
        #     layers.Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
        # self.steps.add(layers.Dense(action_space, activation="tanh"))
        # self.steps.add(layers.Dense(1, activation='sigmoid'))

    def __call__(self, observation):
        _input = np.array(observation, dtype=np.float32)

        actor_dense = self.actor(_input)
        # steps = self.steps(_input)

        mean = actor_dense
        log_std = tf.multiply(self.log_std, 1.0)
        std = tf.exp(log_std)
        dist = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=std)

        # return dist, steps
        return dist
