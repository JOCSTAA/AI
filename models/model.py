import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
main_scope = os.path.join(current_dir, "..\..", "AI")
sys.path.append(main_scope)

import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.layers import Dense, concatenate
import numpy as np
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
from models.transformers.transformer import Transformer


_MODEL_DIMENSION = 20
_WONDER_HIDDEN_DIMENSION = 1024

class BrainModel:
    def __init__(self, eyes_input_shape, ears_input_shape, action_space, sequence_length, learning_rate, epsilon):
        self.model_in_training = ModelInstance(eyes_input_shape, ears_input_shape, action_space, sequence_length,
                                               learning_rate, epsilon)
        self.current_model = ModelInstance(eyes_input_shape, ears_input_shape, action_space, sequence_length,
                                           learning_rate, epsilon)

    def update_model(self):
        """APPLIES LEARNT MODEL ATTRIBUTES TO CURRENT MODEL"""

        self.current_model.actor.set_weights(self.model_in_training.actor.get_weights())
        self.current_model.wonder.set_weights(self.model_in_training.wonder.get_weights())
        self.current_model.critic.set_weights(self.model_in_training.critic.get_weights())


class ModelInstance:
    def __init__(self, eyes_input_shape, ears_input_shape, action_space, sequence_length, learning_rate, epsilon):
        self.eyes_input_shape = eyes_input_shape
        self.ears_input_shape = ears_input_shape
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.sequence_length = sequence_length
        self.actor, self.critic, self.wonder = self.create()

    def create(self):
        """CREATES AND INITIALIZES THE MODELS

        INITIALIZES MODEL SO THAT MODEL ATTRIBUTES CAN BE COPIED
        """

        actor = IndividualModel(self.action_space, self.sequence_length, ppo_loss=True)
        critic = IndividualModel(1, self.sequence_length)
        wonder_output_vector_length = np.prod(list(self.eyes_input_shape)) + np.prod(list(self.ears_input_shape))
        wonder = IndividualModel(wonder_output_vector_length, self.sequence_length, wonder=True)

        actor.compile(optimizer=Adam(learning_rate=self.learning_rate, epsilon=1e-5))
        critic.compile(optimizer=Adam(learning_rate=self.learning_rate, epsilon=1e-5))
        wonder.compile(optimizer=Adam(learning_rate=self.learning_rate, epsilon=1e-5))

        x,y,z = self.eyes_input_shape
        random_eye_array = np.random.rand(x, y, z) * 255.0
        x,y,z = self.ears_input_shape
        random_ear_array = np.random.rand(x, y, z) * 255.0

        eyes_time_sequence_observation = []
        ears_time_sequence_observation = []
        for _ in range(self.sequence_length):
            eyes_time_sequence_observation.append(random_eye_array)
            ears_time_sequence_observation.append(random_ear_array)

        state = (np.array([eyes_time_sequence_observation]), np.array([ears_time_sequence_observation]))

        dist = actor(state)
        action = dist.sample()
        critic(state)
        wonder([state, action])

        return actor, critic, wonder


class IndividualModel(Model):

    def __init__(self,  action_space, sequence_length, wonder=False, ppo_loss=False):
        super(IndividualModel, self).__init__()
        self.model = Transformer(
            input_embedding_length=_MODEL_DIMENSION,
            sequence_length=sequence_length,
            output_vector_length=action_space
        )
        self.ppo_loss = ppo_loss
        self.wonder = wonder
        if wonder:
            self.ffn = Sequential([
                Dense(_WONDER_HIDDEN_DIMENSION),
                Dense(action_space, activation='tanh')
                ])
        if ppo_loss:
            self.log_std = tf.Variable(shape=(1, action_space), trainable=True,
                                   initial_value=tf.zeros(shape=(1, action_space)))

    def __call__(self, inputs):        
        if self.wonder:
            observation, actions = inputs
            model_output = self.model(observation)
            out = concatenate((model_output, actions))
            out = self.ffn(out)

            return out

        if self.ppo_loss:
            model_output = self.model(inputs)

            mean = model_output
            log_std = tf.multiply(self.log_std, 1.0)
            scale = tf.exp(log_std)
            dist = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=scale)

            return dist
        return self.model(inputs)