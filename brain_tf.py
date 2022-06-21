import random
from model_tf import LAV

from memory import memory
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
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
                 epoch=5):
        self.epoch = epoch
        self.action_space = action_space
        self.memory_being_filled = memory()
        self.memory_being_trained = memory()
        self.training = False
        self.alive = True
        self.thread = Thread(target=self.train_thread)
        self.thread.start()
        self.loss_clip_parameter = 0.2
        self.entropy_coefficient = 0.001
        self.value_coefficient = 0.5
        self.minibatch_size = 32

        # self.model_in_training = LAV(6, 2, 6).to(device)
        self.model_in_training = LAV(eyes_input_shape, ears_input_shape, action_space)
        self.current_model = copy.deepcopy(self.model_in_training)
        self.model_in_training.compile(optimizer=Adam(learning_rate=learning_rate))

    # PPO LOSS FUNCTION
    def train_model(self):
        for _ in range(self.epoch):
            for unit in self.memory_being_trained.yield_minibatches(self.minibatch_size):
                eyes, ears, action, discounted_reward, advantage = unit
                state = (eyes, ears)
                with tf.GradientTape(persistent=True) as tape:
                    new_action, value, _, _ = self.model_in_training(state)

                    ratio = tf.math.exp(new_action - action)

                    surrogate1 = ratio * advantage
                    surrogate2 = tf.clip_by_value(ratio, 1 - self.loss_clip_parameter,
                                                  1 + self.loss_clip_parameter) * advantage

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
                print("new_training")
                self.train_model()
                self.training = False

    # USE THE CURRENT MODEL TO PREDICT THE REAL TIME STATES
    # TRY TO KEEP THIS TIME OPTIMIZED, THIS IS DONE IN REAL TIME
    def predict(self, observation):
        eyes = observation[0]
        ears = observation[1]
        state = (np.array([eyes]), np.array([ears]))
        action, value, _, _ = self.current_model(state)

        # discounted_reward = compute_reward() # HAHA DEAL WITH THIS SHIII BOYYYYYYYY!
        discounted_reward = 1
        advantage = discounted_reward - value

        unit = [eyes, ears, action, discounted_reward, advantage]
        self.memory_being_filled.store_in_memory(unit)

        result = action.numpy()[0]
        print(result)
        return result

    # PREVIOUS BATCH IS DONE TRAINING, TRAIN THE NEW BATCH:
    # UPDATE THE MODEL FIRST THOUGH
    # SO THAT THE CURRENT MODEL BEING USED TO PREDICT IS THE LATEST COPY
    def update_model(self):
        self.current_model = copy.deepcopy(self.model_in_training)
        self.memory_being_trained.memory = self.memory_being_filled.copy()

        self.training = True  # Initiates next training thread

        self.memory_being_filled.clear()

    # TERMINATES LEARNING PROCESS
    def kill_brain(self):
        self.alive = False
        self.thread.join()
