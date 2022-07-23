import random
from model_tf import LAV

from memory import memory
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import time
import copy
from threading import Thread

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Brain():
    def __init__(self,
                 eyes_input_shape=(256, 256, 3),  # update to actual eye shape no resize
                 ears_input_shape=(20, 17),  # update to actual ear shape
                 lstm_timestep=10,
                 learning_rate=0.003,
                 action_space=10,
                 threading = True,
                 epoch=5):
        self.epoch = epoch
        self.action_space = action_space
        self.eyes_input_shape = eyes_input_shape
        self.ears_input_shape = ears_input_shape
        self.lstm_timestep = lstm_timestep
        self.memory_being_filled = memory()
        self.memory_being_trained = memory()
        self.training = False
        self.alive = True
        self.threading = threading
        if threading:
            self.thread = Thread(target=self.train_thread)
            self.thread.start()
        self.loss_clip_parameter = 0.2
        self.entropy_coefficient = 0.001
        self.value_coefficient = 0.5
        self.minibatch_size = 32

        self.model_in_training = LAV(eyes_input_shape, ears_input_shape, action_space)
        self.model_in_training.compile(optimizer=Adam(learning_rate=learning_rate))
        self.initialize_model()
        self.current_model = LAV(eyes_input_shape, ears_input_shape, action_space)

    # PPO LOSS FUNCTION
    def train_model(self):
        for _ in range(self.epoch):
            for unit in self.memory_being_trained.yield_minibatches(self.minibatch_size):
                eyes, ears, action_sample, old_log_probs, discounted_reward, advantages = unit
                state = (eyes, ears)
                with tf.GradientTape() as tape:
                    value, dist_new = self.model_in_training(state)

                    new_log_probs = dist_new.log_prob(action_sample)

                    ratio = tf.math.exp(new_log_probs - old_log_probs)

                    surrogate1 = ratio * advantages
                    surrogate2 = tf.clip_by_value(ratio, 1 - self.loss_clip_parameter,
                                                  1 + self.loss_clip_parameter) * advantages

                    actor_loss = -tf.math.minimum(surrogate1, surrogate2)
                    actor_loss = tf.math.reduce_mean(actor_loss)
                    value_loss = keras.losses.MSE(discounted_reward, value)
                    loss = (self.value_coefficient * value_loss) + actor_loss

                params = self.model_in_training.trainable_variables
                grads = tape.gradient(loss, params)
                self.model_in_training.optimizer.apply_gradients(zip(grads, params))

    # THIS FUNCTION IS THE ASYNCHRONOUS FUNCTION THAT WOULD KEEP RUNNING TILL LEARNING PROCESS IS TERMINATED
    # IT INITIATES THE TRAINING OF THE NEW BATCH WHEN SELF.TRAINING IS TOGGLED FROM OFF TO ON
    def train_thread(self):
        while self.alive:
            if self.training:
                self.train_model()
                self.training = False

    # USE THE CURRENT MODEL TO PREDICT THE REAL TIME STATES
    # TRY TO KEEP THIS TIME OPTIMIZED, THIS IS DONE IN REAL TIME
    def predict(self, observation):
        eyes = observation[0]
        ears = observation[1]
        state = (np.array([eyes]), np.array([ears]))
        value, dist = self.current_model(state)

        action_sample = dist.sample()
        log_probs = dist.log_prob(action_sample)

        # discounted_reward = compute_reward() # HAHA DEAL WITH THIS SHIII BOYYYYYYYY!
        discounted_reward = 1
        advantage = discounted_reward - value

        unit = [eyes, ears, action_sample, log_probs, discounted_reward, advantage]
        self.memory_being_filled.store_in_memory(unit)

        result = action_sample.numpy()[0]
        return result

    # PREVIOUS BATCH IS DONE TRAINING, TRAIN THE NEW BATCH:
    # UPDATE THE MODEL FIRST THOUGH
    # SO THAT THE CURRENT MODEL BEING USED TO PREDICT IS THE LATEST COPY
    def update_model(self):
        self.copy_model()
        self.memory_being_trained.memory = self.memory_being_filled.copy()
        self.training = True  # Initiates next training thread

        if not self.threading:
            self.train_model()
            self.training = False

        self.memory_being_filled.clear()

    # TERMINATES LEARNING PROCESS
    def kill_brain(self):
        self.alive = False
        self.thread.join()

    # APPLIES LEARNT MODEL ATTRIBUTES TO CURRENT MODEL
    def copy_model(self):
        self.current_model.set_weights(self.model_in_training.get_weights())

    # INITIALIZES MODEL SO THAT MODEL ATTRIBUTES CAN BE COPIED
    def initialize_model(self):
        random_eye_array = np.random.randint(low=0, high=255, size=self.eyes_input_shape)
        random_ear_array = np.random.randint(low=0, high=255, size=self.ears_input_shape)

        eyes_time_sequence_observation = []
        ears_time_sequence_observation = []
        for _ in range(self.lstm_timestep):
            eyes_time_sequence_observation.append(random_eye_array)
            ears_time_sequence_observation.append(random_ear_array)

        model_input = (np.array([eyes_time_sequence_observation]), np.array([ears_time_sequence_observation]))
        self.model_in_training(model_input)
