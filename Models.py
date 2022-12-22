import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam

class ModelInstance:
    def __init__(self, eyes_input_shape, ears_input_shape, action_space, learning_rate, epsilon, lstm_timestep):
        self.eyes_input_shape = eyes_input_shape
        self.ears_input_shape = ears_input_shape
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.lstm_timestep = lstm_timestep
        self.actor, self.critic, self.wonder = self.create()

    def create(self):
        """CREATES AND INITIALIZES THE MODELS

        INITIALIZES MODEL SO THAT MODEL ATTRIBUTES CAN BE COPIED
        """

        actor = Actor(self.action_space)
        critic = Critic(self.action_space)
        wonder = Wonder(self.eyes_input_shape, self.ears_input_shape, self.action_space)

        actor.compile(optimizer=Adam(learning_rate=self.learning_rate, epsilon=1e-5))
        critic.compile(optimizer=Adam(learning_rate=self.learning_rate, epsilon=1e-5))
        wonder.compile(optimizer=Adam(learning_rate=self.learning_rate, epsilon=1e-5))

        random_eye_array = np.random.randint(low=0, high=255, size=self.eyes_input_shape)
        random_ear_array = np.random.randint(low=0, high=255, size=self.ears_input_shape)

        eyes_time_sequence_observation = []
        ears_time_sequence_observation = []
        for _ in range(self.lstm_timestep):
            eyes_time_sequence_observation.append(random_eye_array)
            ears_time_sequence_observation.append(random_ear_array)

        state = (np.array([eyes_time_sequence_observation]), np.array([ears_time_sequence_observation]))

        dist = actor(state)
        action = dist.sample()
        critic(state)
        wonder(state, action)

        return actor, critic, wonder


class Model:
    def __init__(self, eyes_input_shape, ears_input_shape, action_space, learning_rate, epsilon, lstm_timestep):
        self.model_in_training = ModelInstance(eyes_input_shape, ears_input_shape, action_space,
                                               learning_rate, epsilon, lstm_timestep)
        self.current_model = ModelInstance(eyes_input_shape, ears_input_shape, action_space,
                                           learning_rate, epsilon, lstm_timestep)

    def update_model(self):
        """APPLIES LEARNT MODEL ATTRIBUTES TO CURRENT MODEL"""

        self.current_model.actor.set_weights(self.model_in_training.actor.get_weights())
        self.current_model.wonder.set_weights(self.model_in_training.wonder.get_weights())
        self.current_model.critic.set_weights(self.model_in_training.critic.get_weights())


class Actor(keras.Model):
    def __init__(self, action_space):
        super(Actor, self).__init__()

        # CONVOLUTION LAYERS
        self.actor_eye_conv = keras.Sequential()
        self.actor_eye_conv.add(layers.TimeDistributed(layers.Conv2D(32, 3, activation="relu", padding="same")))
        self.actor_eye_conv.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2), padding="same")))
        self.actor_eye_conv.add(layers.TimeDistributed(layers.Conv2D(64, 5, activation="relu", padding="same")))
        self.actor_eye_conv.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2), padding="same")))
        self.actor_eye_conv.add(layers.TimeDistributed(layers.Flatten()))

        self.actor_ear_conv = keras.Sequential()
        self.actor_ear_conv.add(layers.TimeDistributed(layers.Conv2D(32, 3, activation="relu", padding="same")))
        self.actor_ear_conv.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=2, padding="same")))
        self.actor_ear_conv.add(layers.TimeDistributed(layers.Conv2D(64, 5, activation="relu", padding="same")))
        self.actor_ear_conv.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=2, padding="same")))
        self.actor_ear_conv.add(layers.TimeDistributed(layers.Flatten()))

        # LSTM
        self.actor_lstm1 = layers.LSTM(2, return_sequences=True)
        self.actor_lstm2 = layers.LSTM(10, return_sequences=False)

        # DENSE
        self.actor_dense = keras.Sequential()
        self.actor_dense.add(layers.Dense(64, activation='relu'))
        self.actor_dense.add(layers.Dense(64, activation='relu'))

        # OUTPUT
        self.actor_mean = layers.Dense(action_space, activation='tanh')
        self.log_std = tf.Variable(shape=(1, action_space), trainable=True,
                                   initial_value=tf.zeros(shape=(1, action_space)))

    def __call__(self, observation):
        eye_input = np.array(observation[0], dtype=np.float32)
        ear_input = np.array(observation[1], dtype=np.float32)

        actor_eye_conv, actor_ear_conv = self.actor_eye_conv(eye_input), self.actor_ear_conv(ear_input)
        actor_merged = layers.concatenate((actor_eye_conv, actor_ear_conv))
        actor_lstm1 = self.actor_lstm1(actor_merged)
        actor_lstm2 = self.actor_lstm2(actor_lstm1)
        actor_dense = self.actor_dense(actor_lstm2)

        mean = self.actor_mean(actor_dense)
        log_std = tf.multiply(self.log_std, 1.0)
        scale = tf.exp(log_std)
        dist = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=scale)

        return dist


class Critic(keras.Model):
    def __init__(self, input_shape):
        super(Critic, self).__init__()

        # CONVOLUTION LAYERS
        self.value_eye_conv = keras.Sequential()
        self.value_eye_conv.add(layers.TimeDistributed(layers.Conv2D(32, 3, activation="relu", padding="same")))
        self.value_eye_conv.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2), padding="same")))
        self.value_eye_conv.add(layers.TimeDistributed(layers.Conv2D(64, 5, activation="relu", padding="same")))
        self.value_eye_conv.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2), padding="same")))
        self.value_eye_conv.add(layers.TimeDistributed(layers.Flatten()))

        self.value_ear_conv = keras.Sequential()
        self.value_ear_conv.add(layers.TimeDistributed(layers.Conv2D(32, 3, activation="relu", padding="same")))
        self.value_ear_conv.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=2, padding="same")))
        self.value_ear_conv.add(layers.TimeDistributed(layers.Conv2D(64, 5, activation="relu", padding="same")))
        self.value_ear_conv.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=2, padding="same")))
        self.value_ear_conv.add(layers.TimeDistributed(layers.Flatten()))

        # LSTM
        self.value_lstm1 = layers.LSTM(2, return_sequences=True)
        self.value_lstm2 = layers.LSTM(10, return_sequences=False)

        # DENSE
        self.value_dense = keras.Sequential()
        self.value_dense.add(layers.Dense(64, activation='relu'))
        self.value_dense.add(layers.Dense(64, activation='relu'))

        # OUTPUT
        self.value_output = layers.Dense(1)

    def __call__(self, observation):
        eye_input = np.array(observation[0], dtype=np.float32)
        ear_input = np.array(observation[1], dtype=np.float32)

        value_eye_conv, value_ear_conv = self.value_eye_conv(eye_input), self.value_ear_conv(ear_input)
        value_merged = layers.concatenate((value_eye_conv, value_ear_conv))
        value_lstm1 = self.value_lstm1(value_merged)
        value_lstm2 = self.value_lstm2(value_lstm1)
        value_dense = self.value_dense(value_lstm2)
        value_output = self.value_output(value_dense)

        return value_output


class Wonder(keras.Model):
    """
    Wonder Model:
    Predicts what the new state gotten from performing an action on a given state would be

    Input = ((LSTM batch eyes images, LSTM batch ears images), Action_sample)
    Output = (Single batch eye image, Single batch ear image)
    """
    def __init__(self, eyes_input_shape, ears_input_shape, action_space):
        super(Wonder, self).__init__()
        self.eyes_input_shape = eyes_input_shape
        self.ears_input_shape = ears_input_shape

        # CONVOLUTION LAYERS
        self.wonder_eye_conv = keras.Sequential()
        self.wonder_eye_conv.add(layers.TimeDistributed(layers.Conv2D(32, 3, activation="relu", padding="same")))
        self.wonder_eye_conv.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2), padding="same")))
        self.wonder_eye_conv.add(layers.TimeDistributed(layers.Conv2D(64, 5, activation="relu", padding="same")))
        self.wonder_eye_conv.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2), padding="same")))
        self.wonder_eye_conv.add(layers.TimeDistributed(layers.Flatten()))

        self.wonder_ear_conv = keras.Sequential()
        self.wonder_ear_conv.add(layers.TimeDistributed(layers.Conv2D(32, 3, activation="relu", padding="same")))
        self.wonder_ear_conv.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=2, padding="same")))
        self.wonder_ear_conv.add(layers.TimeDistributed(layers.Conv2D(64, 5, activation="relu", padding="same")))
        self.wonder_ear_conv.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=2, padding="same")))
        self.wonder_ear_conv.add(layers.TimeDistributed(layers.Flatten()))

        # LSTM
        self.wonder_lstm1 = layers.LSTM(2, return_sequences=True)
        self.wonder_lstm2 = layers.LSTM(10, return_sequences=False)

        # DENSE
        self.wonder_dense = keras.Sequential()
        self.wonder_dense.add(layers.Dense(64, activation='relu'))
        self.wonder_dense.add(layers.Dense(64, activation='relu'))

        # OUTPUT
        number_of_eye_neurons = np.prod(list(eyes_input_shape))
        number_of_ear_neurons = np.prod(list(ears_input_shape))
        self.wonder_eyes_output = layers.Dense(number_of_eye_neurons, activation='relu')
        self.wonder_ears_output = layers.Dense(number_of_ear_neurons, activation='relu')

    def __call__(self, observation, action):
        eye_input = np.array(observation[0], dtype=np.float32)
        ear_input = np.array(observation[1], dtype=np.float32)
        action = np.array(action, dtype=np.float32)

        wonder_eye_conv, wonder_ear_conv = self.wonder_eye_conv(eye_input), self.wonder_ear_conv(ear_input)
        wonder_merged = layers.concatenate((wonder_eye_conv, wonder_ear_conv))
        wonder_lstm1 = self.wonder_lstm1(wonder_merged)
        wonder_lstm2 = self.wonder_lstm2(wonder_lstm1)
        wonder_merged = layers.concatenate((wonder_lstm2, action))
        wonder_dense = self.wonder_dense(wonder_merged)
        wonder_eye_output = self.wonder_eyes_output(wonder_dense)
        wonder_ear_output = self.wonder_ears_output(wonder_dense)
        wonder_output = layers.concatenate((wonder_eye_output, wonder_ear_output))

        return wonder_output
