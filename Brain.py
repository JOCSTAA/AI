import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Brain():
    def __init__(self,
                 eyes_input_shape=(256, 256, 3),
                 ears_input_shape=(256, 256, 3),
                 lstm_timestep=10,
                 learning_rate=0.003):
        self.model = self.create_model(eyes_input_shape, ears_input_shape, lstm_timestep, learning_rate)

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
        output = layers.Dense(12, activation='tanh')(dense2)

        # compile model =====================================================================================================
        actor = keras.Model(inputs=[eyes_input, ears_input], outputs=output, name="actor")
        actor.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # INPUT PPO LOSS HERE!!
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
