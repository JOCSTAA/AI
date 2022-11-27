import random
from tesmodel import ActorW, Critic, Wonder

from LAWC_memory import memory
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
                 learning_rate=0.003,
                 epoch=5):
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
        self.wonder_steps = 5
        self.minibatch_size = 256
        self.learning_rate = learning_rate

        self.model_in_training = []
        self.current_model = []
        self.prep_model()

    def prep_model(self):
        actor = ActorW(self.action_space, self.input_shape)
        critic = Critic(self.input_shape)
        wonderer = Wonder(self.input_shape[0])
        actor.compile(optimizer=Adam(learning_rate=self.learning_rate))
        critic.compile(optimizer=Adam(learning_rate=self.learning_rate))
        wonderer.compile(optimizer=Adam(learning_rate=self.learning_rate))
        self.model_in_training = actor, wonderer, critic

        actor = ActorW(self.action_space, self.input_shape)
        critic = Critic(self.input_shape)
        wonderer = Wonder(self.input_shape[0])
        actor.compile(optimizer=Adam(learning_rate=self.learning_rate))
        critic.compile(optimizer=Adam(learning_rate=self.learning_rate))
        wonderer.compile(optimizer=Adam(learning_rate=self.learning_rate))
        self.current_model = actor, wonderer, critic

        self.initialize_model()
        self.update_model()

    # PPO LOSS FUNCTION
    def train_model(self):
        for _ in range(self.epoch):
            for unit in self.memory_being_trained.yield_minibatches(self.minibatch_size):
                state, action_sample, old_log_probs, discounted_reward, advantages, real_state = unit
                state = tf.convert_to_tensor(state)
                action_sample = tf.convert_to_tensor(action_sample)
                old_log_probs = tf.convert_to_tensor(old_log_probs)
                discounted_reward = tf.convert_to_tensor(discounted_reward)
                advantages = tf.convert_to_tensor(advantages)
                real_state = tf.convert_to_tensor(real_state)
                actor, wonderer, critic = self.model_in_training
                print(len(state) , "in_training")

                with tf.GradientTape(persistent=True) as tape:
                    dist_new, value = actor(state), critic(state)
                    action_sample_w = dist_new.sample()
                    action_w = action_sample_w.numpy()
                    new_state = wonderer((state, action_w))

                    new_log_probs = dist_new.log_prob(action_sample)

                    ratio = tf.math.exp(new_log_probs - old_log_probs)

                    surrogate1 = ratio * advantages
                    surrogate2 = tf.clip_by_value(ratio, 1 - self.loss_clip_parameter,
                                                  1 + self.loss_clip_parameter) * advantages

                    actor_loss = -tf.math.minimum(surrogate1, surrogate2)
                    actor_loss = tf.math.reduce_mean(actor_loss)

                    critic_loss = keras.losses.MSE(discounted_reward, value)

                    wonderer_loss = keras.losses.MSE(real_state, new_state)

                actor_params = actor.trainable_variables
                actor_grads = tape.gradient(actor_loss, actor_params)
                actor.optimizer.apply_gradients(zip(actor_grads, actor_params))

                critic_params = critic.trainable_variables
                critic_grads = tape.gradient(critic_loss, critic_params)
                critic.optimizer.apply_gradients(zip(critic_grads, critic_params))

                wonderer_params = wonderer.trainable_variables
                wonderer_grads = tape.gradient(wonderer_loss, wonderer_params)
                wonderer.optimizer.apply_gradients(zip(wonderer_grads, wonderer_params))

    def wonder(self, state):
        actor, wonderer, critic = self.current_model
        top_value = float("-inf")
        for _ in range(self.wonder_steps):
            dist = actor(state)
            action_sample = dist.sample()

            action = action_sample.numpy()

            new_state = wonderer((state, action))
            value_of_new_state = critic(new_state)

            if value_of_new_state > top_value:
                top_value = value_of_new_state
                best_action_sample = action_sample
                log_probs = dist.log_prob(best_action_sample)

        return best_action_sample, log_probs, critic(state)

    # USE THE CURRENT MODEL TO PREDICT THE REAL TIME STATES
    # TRY TO KEEP THIS TIME OPTIMIZED, THIS IS DONE IN REAL TIME
    def predict(self, observation):
        state = np.reshape(observation, [1, self.input_shape[0]])
        action_sample, log_probs, value = self.wonder(state)

        result = action_sample.numpy()[0]

        new_state, reward, done, info = self.env.step(result)

        unit = [state[0], action_sample[0].numpy(), log_probs[0].numpy(), reward, value[0].numpy(), new_state]
        self.memory_being_filled.store_in_memory(unit)

        result = action_sample.numpy()[0]
        return result, new_state, done, reward

    # PREVIOUS BATCH IS DONE TRAINING, TRAIN THE NEW BATCH:
    # UPDATE THE MODEL FIRST THOUGH
    # SO THAT THE CURRENT MODEL BEING USED TO PREDICT IS THE LATEST COPY
    def update_model(self):
        self.copy_model()

        if self.memory_being_filled.memory["state"]:
            # print("training")
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
        self.current_model[2].set_weights(self.model_in_training[2].get_weights())

    # INITIALIZES MODEL SO THAT MODEL ATTRIBUTES CAN BE COPIED
    def initialize_model(self):
        input_ = np.random.rand(self.input_shape[0])
        state = np.reshape(input_, [1, self.input_shape[0]])

        actor, wonderer, critic = self.model_in_training
        dist = actor(state)
        critic(state)
        action_sample = dist.sample()
        action = action_sample.numpy()
        wonderer((state, action))

        actor, wonderer, critic = self.current_model
        dist = actor(state)
        critic(state)
        action_sample = dist.sample()
        action = action_sample.numpy()
        wonderer((state, action))