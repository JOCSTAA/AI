import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K
from threading import Thread


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
                 action_space=10,
                 epoch=5):
        self.epoch = epoch
        self.action_space = action_space
        # self.log_std = -0.5 * np.ones(self.action_space, dtype=np.float32)
        # self.std = np.exp(self.log_std)
        self.actor, self.critic = self.create_model(eyes_input_shape, ears_input_shape, lstm_timestep, learning_rate)
        self.clone_actor, self.clone_critic = self.actor, self.critic
        self.training_batch = []
        self.training = False
        self.alive = True
        self.thread = Thread(target=self.train_helper)
        self.thread.start()

    def actor_ppo_loss(self, y_true, y_pred):
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

    def critic_ppo_loss(self, values):
        def loss(y_true, y_pred):
            LOSS_CLIPPING = 0.2
            clipped_value_loss = values + K.clip(y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING)
            v_loss1 = (y_true - clipped_value_loss) ** 2
            v_loss2 = (y_true - y_pred) ** 2

            value_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))
            # value_loss = K.mean((y_true - y_pred) ** 2) # standard PPO loss
            return value_loss

        return loss

    def create_model(self, eyes_input_shape, ears_input_shape, time_step, learning_rate):
        # ====================================================================================================================
        # INPUT
        # ====================================================================================================================
        input = keras.Input(shape=(2,time_step, 256, 256, 3))
        eyes_input, ears_input = input[:,0] , input[:,1]
        old_values = keras.Input(shape=(1,))

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
        actor = keras.Model(inputs=input, outputs=output, name="actor")
        actor.compile(
            loss=self.actor_ppo_loss,
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
        critic = keras.Model(inputs=input, outputs=output, name="critic")
        critic.compile(
            loss="mse",
            optimizer=keras.optimizers.Adam(lr=learning_rate)
        )

        critic.summary()

        return actor, critic

    def train_helper(self):
        while self.alive:
            if self.training:
                self.clone_actor = tf.keras.models.clone_model(self.actor)
                self.clone_critic = tf.keras.models.clone_model(self.critic)

                states, actions, rewards, predictions = np.vstack(self.training_batch[:, 0]), np.vstack(
                    self.training_batch[:, 1]), np.vstack(
                    self.training_batch[:, 2]), np.vstack(self.training_batch[:, 3])

                values = self.clone_critic.predict(states)  # deal with reward here
                advantages = rewards - values

                y_true = np.hstack([advantages, actions, predictions])
                self.clone_actor.fit(states, y_true, epochs=self.epoch, verbose=0, shuffle=True)
                self.clone_critic.fit(states, rewards, epochs=self.epoch, verbose=0, shuffle=True)

                self.training = False

    def train(self, batch):
        self.training_batch = np.array(batch)
        self.training = True

    def predict(self, observation):
        print(np.array([observation]).shape)
        prediction = self.actor.predict(np.array([observation]))[0]

        print(prediction)
        action = np.random.choice(self.action_space, p=prediction)
        return action, prediction

    def update_model(self):
        self.actor = self.clone_actor
        self.critic = self.clone_critic

    def kill_brain(self):
        self.alive = False
        self.thread.join()


# =======================================================================================================================
# TEST MODEL
# =======================================================================================================================

class LAV(nn.Module):  # LSTM, ACTOR, VALUE
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(LAV, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

        self.apply(self.init_weights)

    def forward(self, x):
        value = self.critic(x)
        mu = self.actor(x)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0., std=0.1)
            nn.init.constant_(m.bias, 0.1)


# Actual model