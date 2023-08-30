import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
main_scope = os.path.join(current_dir, "..\..", "AI")
sys.path.append(main_scope)

from models.model import Model
from memory.memory import Memory

import numpy as np
import tensorflow as tf
import random

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


class Brain:
    def __init__(self,
                 eyes_input_shape=(256, 256, 3),  # update to actual eye shape no resize
                 ears_input_shape=(308, 310, 3),  # update to actual ear shape
                 lstm_timestep=10,
                 learning_rate=0.003,
                 action_space=10,
                 batch_size=256,
                 minibatch_size=16,
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
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.learning_rate = learning_rate
        self.epsilon = 1e-5
        self.in_transit = False
        self.training = False

        if batch_size < minibatch_size:
            raise ValueError("minibatch cannot be greater than batch size")

        self.model = Model(eyes_input_shape, ears_input_shape, action_space, learning_rate, self.epsilon, lstm_timestep)

    def train_model(self):
        """PPO LOSS FUNCTION"""
        self.training = True
        for _ in range(self.epoch):
            for unit in self.memory_being_trained.yield_minibatches(self.minibatch_size):
                eyes, ears, action_sample, old_log_probs, advantages, value, new_eyes, new_ears = unit

                eyes = tf.convert_to_tensor(eyes)
                ears = tf.convert_to_tensor(ears)
                action_sample = tf.convert_to_tensor(action_sample)
                old_log_probs = tf.convert_to_tensor(old_log_probs)
                advantages = tf.convert_to_tensor(advantages)
                old_value = tf.convert_to_tensor(value)

                state = (eyes, ears)
                old_predicted_state = self.convert_image_to_wonder_output(new_eyes, new_ears)

                model = self.model.model_in_training
                actor, critic, wonder = model.actor, model.critic, model.wonder

                with tf.GradientTape(persistent=True) as tape:
                    dist_new, value = actor(state), critic(state)
                    predicted_state = wonder(state, action_sample)

                    new_log_probs = dist_new.log_prob(action_sample)

                    ratio = tf.math.exp(new_log_probs - old_log_probs)

                    surrogate1 = ratio * advantages
                    surrogate2 = tf.clip_by_value(ratio, 1 - self.loss_clip_parameter,
                                                  1 + self.loss_clip_parameter) * advantages

                    actor_loss = -tf.math.minimum(surrogate1, surrogate2)
                    actor_loss = tf.math.reduce_mean(actor_loss)

                    critic_loss = tf.square(value - old_value)
                    critic_loss = 0.5 * tf.reduce_mean(critic_loss)

                    wonder_loss = tf.square(predicted_state - old_predicted_state)
                    wonder_loss = 0.5 * tf.reduce_mean(wonder_loss)

                actor_params = actor.trainable_variables
                actor_grads = tape.gradient(actor_loss, actor_params)
                actor.optimizer.apply_gradients(zip(actor_grads, actor_params))

                critic_params = critic.trainable_variables
                critic_grads = tape.gradient(critic_loss, critic_params)
                critic.optimizer.apply_gradients(zip(critic_grads, critic_params))

                wonder_params = wonder.trainable_variables
                wonder_grads = tape.gradient(wonder_loss, wonder_params)
                wonder.optimizer.apply_gradients(zip(wonder_grads, wonder_params))

        self.training = False
        # USE THE CURRENT MODEL TO PREDICT THE REAL TIME STATES
        # TRY TO KEEP THIS TIME OPTIMIZED, THIS IS DONE IN REAL TIME

    def predict(self, observation):
        """
        USE THE CURRENT MODEL TO PREDICT THE REAL TIME STATES.

        TRY TO KEEP THIS TIME OPTIMIZED, THIS IS DONE IN REAL TIME

        param observation:
        :return: action
        """
        if not self.in_transit:
            eyes = observation[0]
            ears = observation[1]
            state = (np.array([eyes]), np.array([ears]))  # batches up observation, turns it into a one unit batch
            model = self.model.current_model
            value = model.critic(state)
            new_value, dist = self.think(state) # with wonder
            # dist = model.actor(state) # without wonder

            action_sample = dist.sample()
            log_probs = dist.log_prob(action_sample)

            # discounted_reward = compute_reward() # IMPLEMENT REWARD ALGORITHM!
            discounted_reward = 1

            unit = [eyes, ears, action_sample[0].numpy(), log_probs[0].numpy(), discounted_reward, value[0][0].numpy()]
            self.memory_being_filled.store_in_memory(unit)

            action = action_sample.numpy()[0]
            return action

    def predict_test(self, observation, env):
        """
        USE THE CURRENT MODEL TO PREDICT THE REAL TIME STATES.

        TRY TO KEEP THIS TIME OPTIMIZED, THIS IS DONE IN REAL TIME

        param observation:
        :return: action
        """
        if not self.in_transit:
            eyes = observation[0]
            ears = observation[1]
            state = (np.array([eyes]), np.array([ears]))  # batches up observation, turns it into a one unit batch
            model = self.model.current_model
            value = model.critic(state)
            new_value, dist = self.think(state)

            action_sample = dist.sample()
            log_probs = dist.log_prob(action_sample)

            action = action_sample.numpy()[0]

            if action < -0.5:
                actual_action = 0
            elif action > 0.5:
                actual_action = 2
            else:
                actual_action = 1

            reward, _, _ = env.step(actual_action)

            # discounted_reward = compute_reward() # IMPLEMENT REWARD ALGORITHM!
            discounted_reward = reward

            unit = [eyes, ears, action_sample[0].numpy(), log_probs[0].numpy(), discounted_reward, value[0][0].numpy()]
            self.memory_being_filled.store_in_memory(unit)

            return action

    def think(self, observation):
        """
        USE THE CURRENT MODEL TO PREDICT THE REAL TIME STATES TAKING ACCOUNT PREDICTIONS OF FUTURE STATES.

        SIMILAR TO PREDICT BUT USES THE WONDER MODEL, WHICH LEARNS TRANSITIONS, SUMS UP ALL THE PREDICTED ACTIONS AND
        TAKES THE ACTION WITH THE HIGHEST PREDICTED VALUE

        :param observation:
        :return: action
        """

        def get_confidence(model_output) -> float:  # IMPLEMENT CONFIDENCE
            """
            Derives the confidence of a model i.e. how sure the model is that the input will lead to that output
            higher = better chance

            param model_output:
            :return:
            """

            return 0

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
                curr_action = curr_dist.sample()
                new_observation = model.wonder(curr_state, curr_action)
                new_observation = self.convert_wonder_output_to_image(new_observation)

                curr_eye_state, curr_ear_state = curr_state
                # curr_eye_state[0]: first element of batch( batch should only contain one element)

                new_state_eyes = np.delete(curr_eye_state[0], 0, 0)
                new_state_eyes = np.append(new_state_eyes, [new_observation[0][0]], axis=0)

                new_state_ears = np.delete(curr_ear_state[0], 0, 0)
                new_state_ears = np.append(new_state_ears, [new_observation[1][0]], axis=0)

                new_state = ([new_state_eyes], [new_state_ears])

                curr_value = model.critic(new_state)

                certainty = get_confidence(new_state)
                exploration = get_confidence(curr_dist)

                explore_wider = get_random_bool_from_confidence(exploration)
                explore_deeper = get_random_bool_from_confidence(certainty)

                if explore_deeper:
                    values.append((curr_value + helper(new_state)[0], curr_dist))

                else:
                    values.append((curr_value, curr_dist))

            values.sort(key=lambda x: -x[0])
            return values[0]

        model = self.model.current_model
        return helper(observation)

    def convert_wonder_output_to_image(self, wonder_output):
        """
        Takes in flattened one dimension layer of neurons and converts them to eyes and ears images

        :param wonder_output: [batch, neurons]
        :return: eyes[batch, l, w, h], ears[batch, l, w, h] (no lstm layer just single state image)
        """

        def helper(neurons, shape):
            image = np.interp(neurons, (0, 1), (0, 255))
            image = image.astype('uint8')
            image = image.reshape(shape)
            return np.array(image)

        number_of_eye_neurons = np.prod(list(self.eyes_input_shape))  # move outside this so it only runs once
        eyes_batch, ears_batch = [], []
        for arr in wonder_output:
            eye_neurons, ear_neurons = np.split(arr, [number_of_eye_neurons])
            eye_image, ear_image = helper(eye_neurons, self.eyes_input_shape), helper(ear_neurons,
                                                                                      self.ears_input_shape)
            eyes_batch.append(eye_image), ears_batch.append(ear_image)
        return eyes_batch, ears_batch

    def convert_image_to_wonder_output(self, eyes, ears):
        """
        Takes in eyes and ears images and converts them to a flattened one dimension layer of neurons

        :param eyes: [batch, lstm, l , w, h]
        :param ears: [batch, lstm, l , w, h]
        :return: [batch, neurons]
        """
        ret = []
        for index in range(len(eyes)):
            eyes_image, ears_image = eyes[index], ears[index]
            eye_neurons, ear_neurons = eyes_image.flatten(), ears_image.flatten()
            wonder_output = np.concatenate((eye_neurons, ear_neurons))
            ret.append(wonder_output)
        return np.array(ret)

    def update_model(self):
        """
        PREVIOUS BATCH IS DONE TRAINING, TRAIN THE NEW BATCH:
        UPDATE THE MODEL FIRST THOUGH
        SO THAT THE CURRENT MODEL BEING USED TO PREDICT IS THE LATEST COPY
        """
        if len(self.memory_being_filled.memory["eyes"]) >= self.batch_size and not self.training:  # do by batch
            self.in_transit = True
            self.model.update_model()
            # print("Training: Number of Inputs in batch = ", len(self.memory_being_filled.memory["eyes"]))
            self.memory_being_trained.memory = self.memory_being_filled.copy()
            self.memory_being_filled.clear()
            self.in_transit = False
            self.train_model()

    def kill_brain(self):
        """TERMINATES LEARNING PROCESS"""
        self.alive = False
