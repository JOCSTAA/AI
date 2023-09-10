import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
main_scope = os.path.join(current_dir, "..\..", "AI")
sys.path.append(main_scope)

from models.model import BrainModel as Model
from memory.memory import Memory

import numpy as np
import tensorflow as tf
import random
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


class Brain:
    def __init__(self,
                 eyes_input_shape=(256, 256, 3),  # update to actual eye shape no resize
                 ears_input_shape=(308, 310, 3),  # update to actual ear shape
                 sequence_length=10,
                 learning_rate=0.003,
                 action_space=10,
                 batch_size=256,
                 minibatch_size=16,
                 epoch=5,
                 wonder=True):
        self.epoch = epoch
        self.action_space = action_space
        self.eyes_input_shape = eyes_input_shape
        self.ears_input_shape = ears_input_shape
        self.sequence_length = sequence_length
        self.memory_being_filled = Memory(gamma=0.99)
        self.memory_being_trained = Memory(gamma=0.99)
        self.alive = True
        self.wonder = wonder
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

        self.model = Model(eyes_input_shape, ears_input_shape, action_space, sequence_length, learning_rate, self.epsilon)

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
                new_eyes = tf.convert_to_tensor(new_eyes)
                new_ears = tf.convert_to_tensor(new_ears)

                state = (eyes, ears)

                model = self.model.model_in_training
                actor, critic, wonder = model.actor, model.critic, model.wonder

                with tf.GradientTape(persistent=True) as tape:
                    dist_new, value = actor(state), critic(state)

                    new_log_probs = dist_new.log_prob(action_sample)

                    ratio = tf.math.exp(new_log_probs - old_log_probs)

                    surrogate1 = ratio * advantages
                    surrogate2 = tf.clip_by_value(ratio, 1 - self.loss_clip_parameter,
                                                  1 + self.loss_clip_parameter) * advantages

                    actor_loss = -tf.math.minimum(surrogate1, surrogate2)
                    actor_loss = tf.math.reduce_mean(actor_loss)

                    critic_loss = tf.square(value - old_value)
                    critic_loss = 0.5 * tf.reduce_mean(critic_loss)

                actor_params = actor.trainable_variables
                actor_grads = tape.gradient(actor_loss, actor_params)
                actor.optimizer.apply_gradients(zip(actor_grads, actor_params))

                critic_params = critic.trainable_variables
                critic_grads = tape.gradient(critic_loss, critic_params)
                critic.optimizer.apply_gradients(zip(critic_grads, critic_params))

                if self.wonder:
                    old_predicted_state = self.convert_images_to_wonder_output(new_eyes, new_ears) #TODO
                    with tf.GradientTape(persistent=True) as tape:
                        dist_new, value = actor(state), critic(state)
                        predicted_state = wonder([state, action_sample])

                        wonder_loss = tf.square(predicted_state - old_predicted_state)
                        wonder_loss = 0.5 * tf.reduce_mean(wonder_loss)


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
            if self.wonder:
                _, dist = self.think(state) # with wonder #TODO
            else:
                dist = model.actor(state) # without wonder

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
        
        def get_action_likelihood(action_prob_distribution, taken_action) -> float:
            """
            Calculates a measure of how likely the sample is according to the distribution.
            A higher log-likelihood indicates a higher likelihood of the sample being generated by the distribution.
            Higher likelihood = lower chance of exploring other options.

            :param action_prob_distribution: Probability distribution of actions
            :param taken_action: The action that was taken
            :return: prob_to_explore_wider: value from 0 - 1
            """
            log_prob = action_prob_distribution.log_prob(taken_action)
            average_probability = tf.reduce_mean(tf.exp(log_prob)).numpy()

            return average_probability
            return 1 - average_probability # actual return but it currently will while explore_wider() run forever#TODO

        def get_bool_from_probability(random_variable: float) -> bool:
            """
            Returns true or false given a random variable

            param random_variable:
            :return: bool
            """

            return random.random() < random_variable

        def explore_future_actions(curr_state) -> tuple:
            """
            Recursively explores different action paths and outputs the path with the highest predicted reward.

            :param curr_state: The current state
            :return: The best action path's value and the corresponding action distribution
            """
            explore_wider = True
            values = []

            while explore_wider:
                dist = model.actor(curr_state)
                action = model.actor(curr_state).sample()
                new_observation = model.wonder([curr_state, action])
                new_observation = self.convert_wonder_output_to_image(new_observation)

                curr_value = model.critic(new_observation)

                certainty = get_confidence(new_observation)
                exploration = get_action_likelihood(dist, action)

                explore_wider = get_bool_from_probability(exploration)
                explore_deeper = get_bool_from_probability(certainty)

                if explore_deeper:
                    values.append((curr_value + explore_future_actions(new_observation)[0], dist))

                else:
                    values.append((curr_value, dist))

            values.sort(key=lambda x: -x[0])
            return values[0]

        model = self.model.current_model
        return explore_future_actions(observation)
    
    def convert_wonder_output_to_image(self, wonder_output):
        """
        Takes in output of neural network and converts them to eyes and ears images

        :param wonder_output: nn_ouput [batch, sequence, output_vectors]
        :return: eye_images[batch, sequence, l, w, h], ear_images[batch, sequence, l, w, h]
        """
        batch_size, sequence_length, _ = wonder_output.shape
        eyes_vector_size = np.prod(self.eyes_input_shape)

        eye_images = wonder_output[:, :, :eyes_vector_size].reshape((batch_size, sequence_length) + self.eyes_input_shape)
        ear_images = wonder_output[:, :, eyes_vector_size:].reshape((batch_size, sequence_length) + self.ears_input_shape)

        return eye_images, ear_images

    def convert_images_to_wonder_output(self, eyes, ears):
        """
        Takes in eyes and ears images and converts them to a flattened one dimension layer of neurons

        :param eyes: [batch, sequence, l , w, h]
        :param ears: [batch, sequence, l , w, h]
        :return: [batch, sequence, neurons]
        """
        batch, sequence_length, _, _, _ = eyes.shape

        eyes_flattened = eyes.reshape((batch, sequence_length, -1))
        ears_flattened = ears.reshape((batch, sequence_length, -1))
        output_vector = np.concatenate((eyes_flattened, ears_flattened), axis=-1)

        return output_vector

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
