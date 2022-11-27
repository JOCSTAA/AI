import random
from tesmodel import Actor, Critic
from tesmemory import memory

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam

import gym
import pylab
import time
import copy
from threading import Thread

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

class Brain():
    def __init__(self,
                 learning_rate=0.00025,
                 epoch=10):
        self.epoch = epoch
        self.env_name = "BipedalWalker-v3"
        self.env = gym.make(self.env_name)
        self.env.reset()
        self.action_space = self.env.action_space.shape[0]
        self.input_shape = self.env.observation_space.shape
        self.memory_being_filled = memory(gamma = 0.99)
        self.memory_being_trained = memory(gamma= 0.99)
        self.alive = True
        self.loss_clip_parameter = 0.2
        self.entropy_coefficient = 0.001
        self.value_coefficient = 0.5
        self.minibatch_size = 512
        self.learning_rate = learning_rate

        self.model_in_training = []
        self.current_model = []
        self.prep_model()

    def prep_model(self):
        actor = Actor(self.action_space)
        critic = Critic(self.input_shape)
        actor.compile(optimizer=Adam(learning_rate=self.learning_rate, epsilon = 1e-5))
        critic.compile(optimizer=Adam(learning_rate=self.learning_rate, epsilon = 1e-5))
        self.model_in_training = actor, critic

        actor = Actor(self.action_space)
        critic = Critic(self.input_shape)
        actor.compile(optimizer=Adam(learning_rate=self.learning_rate, epsilon = 1e-5))
        critic.compile(optimizer=Adam(learning_rate=self.learning_rate, epsilon = 1e-5))
        self.current_model = actor, critic
        self.initialize_model()
        self.update_model()

    # PPO LOSS FUNCTION
    def train_model(self):
        for _ in range(self.epoch):
            for unit in self.memory_being_trained.yield_minibatches(self.minibatch_size):
                state, action_sample, old_log_probs, advantages, target, value = unit

                state = tf.convert_to_tensor(state)
                action_sample = tf.convert_to_tensor(action_sample)
                old_log_probs = tf.convert_to_tensor(old_log_probs)
                advantages = tf.convert_to_tensor(advantages)
                target = tf.convert_to_tensor(target)
                old_value = tf.convert_to_tensor(value)
                # print(len(state), "in_training")
                actor, critic = self.model_in_training

                with tf.GradientTape(persistent=True) as tape:
                    dist_new, value = actor(state), critic(state)

                    new_log_probs = dist_new.log_prob(action_sample)

                    ratio = tf.math.exp(new_log_probs - old_log_probs)

                    surrogate1 = ratio * advantages
                    surrogate2 = tf.clip_by_value(ratio, 1 - self.loss_clip_parameter,
                                                  1 + self.loss_clip_parameter) * advantages

                    actor_loss = -tf.math.minimum(surrogate1, surrogate2)
                    actor_loss = tf.math.reduce_mean(actor_loss)

                    clipped_value_loss = old_value + tf.clip_by_value(value - old_value, -self.loss_clip_parameter, self.loss_clip_parameter)
                    v_loss1 = tf.square(target - clipped_value_loss)
                    v_loss2 = tf.square(target - value)


                    # critic_loss = keras.losses.MSE(value, target)
                    critic_loss = 0.5 * tf.reduce_mean(tf.maximum(v_loss1, v_loss2))

                    entropy = dist_new.entropy()
                    entropy_loss = tf.reduce_mean(entropy)

                actor_params = actor.trainable_variables
                actor_grads = tape.gradient(actor_loss, actor_params)
                actor.optimizer.apply_gradients(zip(actor_grads, actor_params))

                critic_params = critic.trainable_variables
                critic_grads = tape.gradient(critic_loss, critic_params)
                critic.optimizer.apply_gradients(zip(critic_grads, critic_params))


    # USE THE CURRENT MODEL TO PREDICT THE REAL TIME STATES
    # TRY TO KEEP THIS TIME OPTIMIZED, THIS IS DONE IN REAL TIME
    def predict(self, observation):
        state = np.reshape(observation, [1, self.input_shape[0]])
        actor, critic = self.current_model
        value, dist = critic(state), actor(state)

        action_sample = dist.sample()
        log_probs = dist.log_prob(action_sample)

        result = action_sample.numpy()[0]

        new_state, reward, done, info = self.env.step(result)
        new_state = np.reshape(new_state, [1, self.input_shape[0]])
        new_value = critic(new_state)

        unit = [state[0], action_sample[0].numpy(), log_probs[0].numpy(), reward, value[0][0].numpy(), new_value[0][0].numpy(), done]
        self.memory_being_filled.store_in_memory(unit)

        return result, new_state, done, reward

    # PREVIOUS BATCH IS DONE TRAINING, TRAIN THE NEW BATCH:
    # UPDATE THE MODEL FIRST THOUGH
    # SO THAT THE CURRENT MODEL BEING USED TO PREDICT IS THE LATEST COPY
    def update_model(self):
        if len(self.memory_being_filled.memory["state"]) >= self.minibatch_size:# do by batch
        # if self.memory_being_filled.memory["state"]:
            self.copy_model()
            print("training", len(self.memory_being_filled.memory["state"]))
            self.memory_being_trained.memory = self.memory_being_filled.copy()
            self.memory_being_filled.clear()
            self.train_model()

    # TERMINATES LEARNING PROCESS
    def kill_brain(self):
        self.alive = False

    # APPLIES LEARNT MODEL ATTRIBUTES TO CURRENT MODEL
    def copy_model(self):
        self.current_model[0].set_weights(self.model_in_training[0].get_weights())
        self.current_model[1].set_weights(self.model_in_training[1].get_weights())

    # INITIALIZES MODEL SO THAT MODEL ATTRIBUTES CAN BE COPIED
    def initialize_model(self):
        input_ = np.random.rand(self.input_shape[0])
        state = np.reshape(input_, [1, self.input_shape[0]])
        actor, critic = self.model_in_training
        actor(state)
        critic(state)
        actor, critic = self.current_model
        actor(state)
        critic(state)