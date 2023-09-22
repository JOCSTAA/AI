import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
main_scope = os.path.join(current_dir, "..\..", "AI")
sys.path.append(main_scope)

from models import model_utils
from memory.memory import Memory
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, concatenate
import numpy as np
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
from models.transformers.transformer import Transformer
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


_MODEL_DIMENSION = 20
_WONDER_HIDDEN_DIMENSION = 1024
_TRAINING_BATCH_SIZE = 5
_EPOCH = 10
_MIN_BATCH_REQUIRED_TO_TRAIN = 5
_EPSILON = 1e-5
_LEARNING_RATE=0.003
_GAMMA=0.99

class BrainModel:
    def __init__(self, 
                 eyes_input_shape, 
                 ears_input_shape, 
                 action_space, 
                 sequence_length, 
                 learning_rate=_LEARNING_RATE, 
                 epsilon=_EPSILON,
                 add_wonder=True,
                 reward_function=None,
                 min_batch_required_to_train=_MIN_BATCH_REQUIRED_TO_TRAIN,
                 epoch=_EPOCH,
                 training_batch_size=_TRAINING_BATCH_SIZE
                 ):
        self.add_wonder=add_wonder
        self.reward_function=reward_function
        self.min_batch_required_to_train=min_batch_required_to_train
        self.epoch=epoch
        self.eyes_input_shape=eyes_input_shape
        self.ears_input_shape=ears_input_shape
        
        if not training_batch_size:
            self.training_batch_size=min_batch_required_to_train
        else:
            self.training_batch_size=training_batch_size
        if min_batch_required_to_train < training_batch_size:
            raise ValueError("training batch size per epoch should be less than minimum batch size required to train, training batch size given {training_batch_size}, minimum batch amount required to train {min_batch_required_to_train}")

        self.model_in_training = ModelInstance(eyes_input_shape, ears_input_shape, action_space, sequence_length,
                                               learning_rate, epsilon)
        self.current_model = ModelInstance(eyes_input_shape, ears_input_shape, action_space, sequence_length,
                                           learning_rate, epsilon)
        
        self.memory = Memory(minibatch_size=self.training_batch_size, 
                             gamma=_GAMMA)

    def update_model(self):
        model_utils.update_model(self)

    def predict(self, observation):
        prediction = model_utils.predict(self, observation)
        return prediction
    
    def train_model_from_saved_data(self):
        pass

    def save_model(self, file_location=None):
        model_utils.save_model(self, file_location)

    def load_model_from_save(self, file_location=None):
        model_utils.load_model(self, file_location)


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

        state = model_utils.instantiate_model_state(self)

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
        # wonder        
        if self.wonder:
            observation, actions = inputs
            model_output = self.model(observation)
            out = concatenate((model_output, actions))
            out = self.ffn(out)

            return out

        # actor
        if self.ppo_loss:
            model_output = self.model(inputs)

            mean = model_output
            log_std = tf.multiply(self.log_std, 1.0)
            scale = tf.exp(log_std)
            dist = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=scale)

            return dist
        
        # critic
        out = tf.reduce_mean(self.model(inputs))
        return out