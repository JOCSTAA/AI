from Models import Model
from Memory import Memory

import numpy as np
import tensorflow as tf
import random


class Brain:
    def __init__(self,
                 eyes_input_shape=(256, 256, 3),  # update to actual eye shape no resize
                 ears_input_shape=(308, 310, 3),  # update to actual ear shape
                 lstm_timestep=10,
                 learning_rate=0.003,
                 action_space=10,
                 epoch=5):
        self.epoch = epoch
        self.action_space = action_space
        self.eyes_input_shape = eyes_input_shape
        self.ears_input_shape = ears_input_shape
        self.lstm_timestep = lstm_timestep
        self.memory_being_filled = Memory(gamma=0.99)
        self.memory_being_trained = Memory(gamma=0.99)
        self.alive = True
        self.loss_clip_parameter = 0.2
        self.entropy_coefficient = 0.001
        self.value_coefficient = 0.5
        self.minibatch_size = 16
        self.learning_rate = learning_rate
        self.epsilon = 1e-5

        self.model = Model(eyes_input_shape, ears_input_shape, action_space, learning_rate, self.epsilon, lstm_timestep)

    def train_model(self):
        """PPO LOSS FUNCTION"""
        for _ in range(self.epoch):
            for unit in self.memory_being_trained.yield_minibatches(self.minibatch_size):
                eyes, ears, action_sample, old_log_probs, advantages, value = unit

                eyes = tf.convert_to_tensor(eyes)
                ears = tf.convert_to_tensor(ears)
                state = (eyes, ears)
                action_sample = tf.convert_to_tensor(action_sample)
                old_log_probs = tf.convert_to_tensor(old_log_probs)
                advantages = tf.convert_to_tensor(advantages)
                old_value = tf.convert_to_tensor(value)
                model = self.model.model_in_training
                actor, critic = model.actor, model.critic

                with tf.GradientTape(persistent=True) as tape:
                    dist_new, value = actor(state), critic(state)

                    new_log_probs = dist_new.log_prob(action_sample)

                    ratio = tf.math.exp(new_log_probs - old_log_probs)

                    surrogate1 = ratio * advantages
                    surrogate2 = tf.clip_by_value(ratio, 1 - self.loss_clip_parameter,
                                                  1 + self.loss_clip_parameter) * advantages

                    actor_loss = -tf.math.minimum(surrogate1, surrogate2)
                    actor_loss = tf.math.reduce_mean(actor_loss)

                    v_loss = tf.square(value - old_value)

                    critic_loss = 0.5 * tf.reduce_mean(v_loss)

                actor_params = actor.trainable_variables
                actor_grads = tape.gradient(actor_loss, actor_params)
                actor.optimizer.apply_gradients(zip(actor_grads, actor_params))

                critic_params = critic.trainable_variables
                critic_grads = tape.gradient(critic_loss, critic_params)
                critic.optimizer.apply_gradients(zip(critic_grads, critic_params))

        # USE THE CURRENT MODEL TO PREDICT THE REAL TIME STATES
        # TRY TO KEEP THIS TIME OPTIMIZED, THIS IS DONE IN REAL TIME

    def predict(self, observation):
        """
        USE THE CURRENT MODEL TO PREDICT THE REAL TIME STATES.

        TRY TO KEEP THIS TIME OPTIMIZED, THIS IS DONE IN REAL TIME

        param observation:
        :return: action
        """
        eyes = observation[0]
        ears = observation[1]
        state = (np.array([eyes]), np.array([ears]))
        model = self.model.current_model
        value = model.critic(state)
        new_value, dist = self.think(state)

        action_sample = dist.sample()
        log_probs = dist.log_prob(action_sample)

        # discounted_reward = compute_reward() # IMPLEMENT REWARD ALGORITHM!
        discounted_reward = 1

        unit = [eyes, ears, action_sample[0].numpy(), log_probs[0].numpy(), discounted_reward, value[0][0].numpy()]
        self.memory_being_filled.store_in_memory(unit)

        action = action_sample.numpy()[0]
        return action

    def think(self, observation):
        """
        USE THE CURRENT MODEL TO PREDICT THE REAL TIME STATES TAKING ACCOUNT PREDICTIONS OF FUTURE STATES.

        SIMILAR TO PREDICT BUT USES THE WONDER MODEL, WHICH LEARNS TRANSITIONS, SUMS UP ALL THE PREDICTED ACTIONS AND
        TAKES THE ACTION WITH THE HIGHEST PREDICTED VALUE

        :param observation:
        :return: action
        """

        def get_confidence(model_output) -> float: # IMPLEMENT CONFIDENCE
            """
            Derives the confidence of a model i.e. how sure the model is that the input will lead to that output

            param model_output:
            :return:
            """

            return 0.5

        def get_random_bool_from_confidence(random_variable: float) -> bool:
            """
            Returns true or false given a random variable which is the confidence of that model

            param random_variable:
            :return:
            """

            return random.random() < random_variable

        def helper(curr_state) -> tuple:
            """
            Recursively explores different action paths and output the path with the highest predicted reward

            param curr_state:
            :return:
            """
            explore_wider = True
            values = []

            while explore_wider:
                curr_dist = model.actor(curr_state)
                curr_action = curr_dist.sample().numpy()[0]
                new_state = model.wonder(curr_state, curr_action)
                curr_value = model.value(new_state)

                certainty = get_confidence(new_state)
                exploration = get_confidence(curr_dist)

                explore_wider = get_random_bool_from_confidence(exploration)
                explore_deeper = get_random_bool_from_confidence(certainty)

                if explore_deeper:
                    values.append((curr_value + helper(new_state), curr_dist))

                else:
                    values.append((curr_value, curr_dist))

            values.sort(key=lambda x: x[0])
            return values[0]

        model = self.model.current_model
        return helper(observation)

    def update_model(self):
        """
        PREVIOUS BATCH IS DONE TRAINING, TRAIN THE NEW BATCH:
        UPDATE THE MODEL FIRST THOUGH
        SO THAT THE CURRENT MODEL BEING USED TO PREDICT IS THE LATEST COPY
        """
        if len(self.memory_being_filled.memory["eyes"]) >= self.minibatch_size:  # do by batch
            self.model.update_model()
            print("training", len(self.memory_being_filled.memory["eyes"]))
            self.memory_being_trained.memory = self.memory_being_filled.copy()
            self.memory_being_filled.clear()
            self.train_model()

    def kill_brain(self):
        """TERMINATES LEARNING PROCESS"""
        self.alive = False
