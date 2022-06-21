import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import tensorflow_probability as tfp


class LAV(keras.Model):
    def __init__(self, eyes_input_shape, ears_input_shape, action_space):
        super(LAV, self).__init__()

        # CONVOLUTION LAYERS
        self.actor_eye_conv = keras.Sequential()
        self.actor_eye_conv.add(layers.TimeDistributed(layers.Conv2D(32, 3, activation="relu", padding="same")))
        self.actor_eye_conv.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2), padding="same")))
        self.actor_eye_conv.add(layers.TimeDistributed(layers.Conv2D(64, 5, activation="relu", padding="same")))
        self.actor_eye_conv.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2), padding="same")))
        self.actor_eye_conv.add(layers.TimeDistributed(layers.Flatten()))

        self.actor_ear_conv = keras.Sequential()
        self.actor_ear_conv.add(layers.TimeDistributed(layers.Conv1D(32, 3, activation="relu", padding="same")))
        self.actor_ear_conv.add(layers.TimeDistributed(layers.MaxPooling1D(pool_size=2, padding="same")))
        self.actor_ear_conv.add(layers.TimeDistributed(layers.Conv1D(64, 5, activation="relu", padding="same")))
        self.actor_ear_conv.add(layers.TimeDistributed(layers.MaxPooling1D(pool_size=2, padding="same")))
        self.actor_ear_conv.add(layers.TimeDistributed(layers.Flatten()))

        self.value_eye_conv = keras.Sequential()
        self.value_eye_conv.add(layers.TimeDistributed(layers.Conv2D(32, 3, activation="relu", padding="same")))
        self.value_eye_conv.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2), padding="same")))
        self.value_eye_conv.add(layers.TimeDistributed(layers.Conv2D(64, 5, activation="relu", padding="same")))
        self.value_eye_conv.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2), padding="same")))
        self.value_eye_conv.add(layers.TimeDistributed(layers.Flatten()))

        self.value_ear_conv = keras.Sequential()
        self.value_ear_conv.add(layers.TimeDistributed(layers.Conv1D(32, 3, activation="relu", padding="same")))
        self.value_ear_conv.add(layers.TimeDistributed(layers.MaxPooling1D(pool_size=2, padding="same")))
        self.value_ear_conv.add(layers.TimeDistributed(layers.Conv1D(64, 5, activation="relu", padding="same")))
        self.value_ear_conv.add(layers.TimeDistributed(layers.MaxPooling1D(pool_size=2, padding="same")))
        self.value_ear_conv.add(layers.TimeDistributed(layers.Flatten()))

        # LSTM
        self.actor_lstm1 = layers.LSTM(10, return_sequences=True)
        self.actor_lstm2 = layers.LSTM(10, return_sequences=False)

        self.value_lstm1 = layers.LSTM(10, return_sequences=True)
        self.value_lstm2 = layers.LSTM(10, return_sequences=False)

        # DENSE
        self.actor_dense = keras.Sequential()
        self.actor_dense.add(layers.Dense(64, activation='relu'))
        self.actor_dense.add(layers.Dense(64, activation='relu'))

        self.value_dense = keras.Sequential()
        self.value_dense.add(layers.Dense(64, activation='relu'))
        self.value_dense.add(layers.Dense(64, activation='relu'))

        # OUTPUT
        self.actor_output = layers.Dense(action_space, activation='tanh')

        self.actor_mean = layers.Dense(action_space, activation='tanh')
        self.actor_log_std = layers.Dense(action_space, activation='tanh')

        self.value_output = layers.Dense(1)

        self.log_std = tf.Variable(shape=(1, action_space), trainable=True, initial_value=tf.zeros(shape=(1, action_space)))

    def __call__(self, observation):
        eye_input = np.array(observation[0], dtype=np.float32)
        ear_input = np.array(observation[1], dtype=np.float32)
        print(eye_input.shape, ear_input.shape)

        value_eye_conv, value_ear_conv = self.value_eye_conv(eye_input), self.value_ear_conv(ear_input)
        value_merged = layers.concatenate((value_eye_conv, value_ear_conv))
        value_lstm1 = self.value_lstm1(value_merged)
        value_lstm2 = self.value_lstm2(value_lstm1)
        value_dense = self.value_dense(value_lstm2)
        value_output = self.value_output(value_dense)

        actor_eye_conv, actor_ear_conv = self.actor_eye_conv(eye_input), self.actor_ear_conv(ear_input)
        actor_merged = layers.concatenate((actor_eye_conv, actor_ear_conv))
        actor_lstm1 = self.actor_lstm1(actor_merged)
        actor_lstm2 = self.actor_lstm2(actor_lstm1)
        actor_dense = self.actor_dense(actor_lstm2)

        # actor_output = self.actor_output(actor_dense)

        # mean = self.actor_mean(actor_dense)
        # log_std = self.actor_log_std(actor_dense)
        # dist = tfp.distributions.MultiVariateNormalDiag(mean=mean, logstd=log_std)

        mean = self.actor_mean(actor_dense)
        log_std = tf.multiply(self.log_std, 1.0)
        scale = tf.exp(log_std)
        dist = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=scale)

        return value_output, dist