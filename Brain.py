import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K


def gaussian_likelihood(actions, pred):
    log_std = -0.5 * np.ones(6, dtype=np.float32)
    pre_sum = -0.5 * (((actions - pred) / (K.exp(log_std) + 1e-8)) ** 2 + 2 * log_std + K.log(2 * np.pi))
    return K.sum(pre_sum, axis=1)


class Brain():
    def __init__(self,
                 eyes_input_shape=(256, 256, 3),
                 ears_input_shape=(256, 256, 3),
                 lstm_timestep=10,
                 learning_rate=0.003,
                 action_space=10):
        self.action_space = action_space
        self.actor, self.critic = self.create_model(eyes_input_shape, ears_input_shape, lstm_timestep, learning_rate)

    def ppo_loss(self, y_true, y_pred):
        advantages, actions, old_log_probs, = y_true[:, :1], y_true[:, 1:1 + self.action_space], y_true[:,
                                                                                                 1 + self.action_space]
        loss_clipping_parameter = 0.2
        new_log_probs = gaussian_likelihood(actions, y_pred)

        ratio = K.exp(new_log_probs - old_log_probs)

        p1 = ratio * advantages
        p2 = tf.where(advantages > 0, (1.0 + loss_clipping_parameter) * advantages,
                      (1.0 - loss_clipping_parameter) * advantages)  # minimum advantage

        actor_loss = -K.mean(K.minimum(p1, p2))

        return actor_loss

    def create_model(self, eyes_input_shape, ears_input_shape, time_step, learning_rate):
        # ====================================================================================================================
        # INPUT
        # ====================================================================================================================
        eyes_input_shape = tuple([time_step] + list(eyes_input_shape))
        eyes_input = keras.Input(
            shape=eyes_input_shape)  # example: shape = (10, 256, 256, 3) for a 10 sequential time 256 by 256 RGB image

        ears_input_shape = tuple([time_step] + list(ears_input_shape))
        ears_input = keras.Input(
            shape=ears_input_shape)  # example: shape = (10, 256, 256, 3) for a 10 sequential time 256 by 256 RGB image

        # ====================================================================================================================
        # ACTOR
        # ====================================================================================================================
        # EYES===============================================================================================================
        eyes_conv_neurons = layers.TimeDistributed(layers.Conv2D(32, 3, activation="relu", padding="same"))(eyes_input)
        eyes_conv_neurons = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2), padding="same"))(
            eyes_conv_neurons)
        eyes_conv_neurons = layers.TimeDistributed(layers.Conv2D(64, 5, activation="relu", padding="same"))(
            eyes_conv_neurons)
        eyes_conv_neurons = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2), padding="same"))(
            eyes_conv_neurons)

        # EARS===============================================================================================================
        ears_conv_neurons = layers.TimeDistributed(layers.Conv2D(32, 3, activation="relu", padding="same"))(ears_input)
        ears_conv_neurons = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2), padding="same"))(
            ears_conv_neurons)
        ears_conv_neurons = layers.TimeDistributed(layers.Conv2D(64, 5, activation="relu", padding="same"))(
            ears_conv_neurons)
        ears_conv_neurons = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2), padding="same"))(
            ears_conv_neurons)

        # MERGE EARS AND EYES NEURONS
        eyes_conv_neurons = layers.TimeDistributed(layers.Flatten())(eyes_conv_neurons)
        ears_conv_neurons = layers.TimeDistributed(layers.Flatten())(ears_conv_neurons)
        merged = layers.Concatenate()([eyes_conv_neurons, ears_conv_neurons])

        # LSTM===============================================================================================================
        lstm1 = layers.LSTM(10, return_sequences=True)(merged)
        lstm2 = layers.LSTM(10, return_sequences=False)(lstm1)

        # DENSE==============================================================================================================
        dense1 = layers.Dense(64, activation='relu')(lstm2)
        dense2 = layers.Dense(64, activation='relu')(dense1)

        # OUTPUT=============================================================================================================
        output = layers.Dense(self.action_space, activation='tanh')(dense2)

        # compile model =====================================================================================================
        actor = keras.Model(inputs=[eyes_input, ears_input], outputs=output, name="actor")
        actor.compile(
            loss=self.ppo_loss,
            optimizer=keras.optimizers.Adam(lr=learning_rate)
        )

        actor.summary()

        # ====================================================================================================================
        # CRITIC
        # ====================================================================================================================
        # EYES===============================================================================================================
        eyes_conv_neurons = layers.TimeDistributed(layers.Conv2D(32, 3, activation="relu", padding="same"))(eyes_input)
        eyes_conv_neurons = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2), padding="same"))(
            eyes_conv_neurons)
        eyes_conv_neurons = layers.TimeDistributed(layers.Conv2D(64, 5, activation="relu", padding="same"))(
            eyes_conv_neurons)
        eyes_conv_neurons = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2), padding="same"))(
            eyes_conv_neurons)

        # EARS===============================================================================================================
        ears_conv_neurons = layers.TimeDistributed(layers.Conv2D(32, 3, activation="relu", padding="same"))(ears_input)
        ears_conv_neurons = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2), padding="same"))(
            ears_conv_neurons)
        ears_conv_neurons = layers.TimeDistributed(layers.Conv2D(64, 5, activation="relu", padding="same"))(
            ears_conv_neurons)
        ears_conv_neurons = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2), padding="same"))(
            ears_conv_neurons)

        # MERGE EARS AND EYES NEURONS
        eyes_conv_neurons = layers.TimeDistributed(layers.Flatten())(eyes_conv_neurons)
        ears_conv_neurons = layers.TimeDistributed(layers.Flatten())(ears_conv_neurons)
        merged = layers.Concatenate()([eyes_conv_neurons, ears_conv_neurons])

        # LSTM===============================================================================================================
        lstm1 = layers.LSTM(10, return_sequences=True)(merged)
        lstm2 = layers.LSTM(10, return_sequences=False)(lstm1)

        # DENSE==============================================================================================================
        dense1 = layers.Dense(64, activation='relu')(lstm2)
        dense2 = layers.Dense(64, activation='relu')(dense1)

        # OUTPUT=============================================================================================================
        output = layers.Dense(1)(dense2)

        # compile model =====================================================================================================
        critic = keras.Model(inputs=[eyes_input, ears_input], outputs=output, name="critic")
        critic.compile(
            loss="mse",
            optimizer=keras.optimizers.Adam(lr=learning_rate)
        )

        critic.summary()

        return actor, critic

brain = Brain()
