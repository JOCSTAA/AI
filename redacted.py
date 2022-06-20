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

import random
from model import LaWV, LaV

import torch
import torch.optim as optim
import copy
from threading import Thread

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class Brain():
    def __init__(self,
                 eyes_input_shape=(256, 256, 3),  # update to actual eye shape no resize
                 ears_input_shape=(20, 17),  # update to actual ear shape
                 lstm_timestep=10,
                 learning_rate=0.003,
                 action_space=10,
                 epoch=5):
        self.epoch = epoch
        self.action_space = action_space
        self.batch_being_filled = []
        self.batch_being_trained = []
        self.training = False
        self.alive = True
        self.thread = Thread(target=self.train_thread)
        self.thread.start()
        self.loss_clip_parameter = 0.2
        self.entropy_coefficient = 0.001
        self.value_coefficient = 0.5

        # self.model_in_training = LaWV(eyes_input_shape, ears_input_shape, action_space, lstm_timestep=lstm_timestep).to(
        #     device)
        # self.current_model = copy.deepcopy(self.model_in_training)
        # self.optimizer = optim.Adam(self.model_in_training.parameters(), lr=learning_rate)

        self.model_in_training = LaV(eyes_input_shape, ears_input_shape, action_space, lstm_timestep=lstm_timestep)
        self.model_in_training.actor.to(device)
        self.model_in_training.value.to(device)
        self.current_actor_model = copy.deepcopy(self.model_in_training.actor)
        self.current_value_model = copy.deepcopy(self.model_in_training.value)
        self.actor_optimizer = optim.Adam(self.model_in_training.actor.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.model_in_training.value.parameters(), lr=learning_rate)

    # PPO LOSS FUNCTION
    def train_model(self):
        random.shuffle(self.batch_being_trained)
        for _ in range(self.epoch):
            for unit in self.batch_being_trained:
                state, action, old_log_probabilities, discounted_reward, advantage = unit
                # dist, value = self.model_in_training(state)

                dist = self.model_in_training.actor(state)
                value = self.model_in_training.value(state)

                advantage = advantage.detach()
                old_log_probabilities = old_log_probabilities.detach()
                value = value.detach()

                entropy = dist.entropy().mean()
                new_log_probability = dist.log_prob(action)

                ratio = (new_log_probability - old_log_probabilities).exp()
                surrogate1 = ratio * advantage
                surrogate2 = torch.clamp(ratio, 1.0 - self.loss_clip_parameter,
                                    1.0 + self.loss_clip_parameter) * advantage

                actor_loss = - torch.min(surrogate1, surrogate2).mean()
                value_loss = (discounted_reward - value).pow(2).mean()

                # loss = (self.value_coefficient * value_loss) + actor_loss - (self.entropy_coefficient * entropy)
                #
                # self.optimizer.zero_grad()
                # loss.backward()
                # self.optimizer.step()

                actor_loss = actor_loss - (self.entropy_coefficient * entropy)
                self.actor_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                actor_loss.backward()
                value_loss.backward()
                self.actor_optimizer.step()
                self.value_optimizer.step()

    # THIS FUNCTION IS THE ASYNCHRONOUS FUNCTION THAT WOULD KEEP RUNNING TILL LEARNING PROCESS IS TERMINATED
    # IT INITIATES THE TRAINING OF THE NEW BATCH WHEN SELF.TRAINING IS TOGGLED FROM OFF TO ON
    def train_thread(self):
        while self.alive:
            if self.training:
                print("new_training")
                self.train_model()
                self.training = False

    # USE THE CURRENT MODEL TO PREDICT THE REAL TIME STATES
    # TRY TO KEEP THIS TIME OPTIMIZED, THIS IS DONE IN REAL TIME
    def predict(self, observation):
        # dist, value = self.current_model(observation)

        dist = self.current_actor_model(observation)
        value = self.current_value_model(observation)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        # discounted_reward = compute_reward() # HAHA DEAL WITH THIS SHIII BOYYYYYYYY!
        discounted_reward = 1
        advantage = discounted_reward - value

        unit = [observation, action, log_prob, discounted_reward, advantage]
        self.batch_being_filled.append(unit)
        result = action.cpu().detach().numpy()[0]
        return result

    # PREVIOUS BATCH IS DONE TRAINING, TRAIN THE NEW BATCH:
    # UPDATE THE MODEL FIRST THOUGH
    # SO THAT THE CURRENT MODEL BEING USED TO PREDICT IS THE LATEST COPY
    def update_model(self):
        # self.current_model = copy.deepcopy(self.model_in_training)

        self.current_actor_model = copy.deepcopy(self.model_in_training.actor)
        self.current_value_model = copy.deepcopy(self.model_in_training.value)

        self.batch_being_trained = self.batch_being_filled.copy()
        self.training = True  # Initiates next training thread

        self.batch_being_filled = []

    # TERMINATES LEARNING PROCESS
    def kill_brain(self):
        self.alive = False
        self.thread.join()


class Actor(nn.Module):
    def __init__(self, eyes_input_shape, ears_input_shape, action_space, lstm_timestep=10, std=0.0):
        super(Actor, self).__init__()
        self.lstm_timestep = lstm_timestep
        # Define hyper parameters for models (example lstm_hidden layers, conv filters ...)

        # CONVOLUTION LAYERS
        self.actor_eyes_conv = nn.Sequential(
            nn.Conv2d(eyes_input_shape[0], 32, (3, 3), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1),
            nn.Conv2d(32, 64, (5, 5), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1),
            nn.Flatten(),
        )

        self.actor_ears_conv = nn.Sequential(
            nn.Conv1d(ears_input_shape[0], 32, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(3, padding=1),
            nn.Conv1d(32, 64, 5, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(3, padding=1),
            nn.Flatten()
        )

        # LSTM
        # the input size should change depending on the num of features
        # ex: input of lstm = (batch, lstm_timestep, num_of_features), then input size = num_of_features
        self.actor_lstm = nn.LSTM(input_size=8448, hidden_size=10, num_layers=5, batch_first=True)

        # DENSE
        self.actor_dense = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # OUTPUT
        self.actor_output = nn.Sequential(
            nn.Linear(64, action_space),
            nn.Tanh()
        )

        # UTILS
        self.log_std = nn.Parameter(torch.ones(1, action_space) * std)
        self.apply(self.init_weights)

    def forward(self, observation):  # Works on single observation not batch
        actor_lstm_input = []
        for observation_frame in observation:
            eye_input, ear_input = torch.FloatTensor([observation_frame[0]]), torch.FloatTensor([observation_frame[1]])
            actor_eye_conv, actor_ear_conv = self.actor_eyes_conv(eye_input), self.actor_ears_conv(ear_input)

            actor_merged = torch.cat((actor_eye_conv, actor_ear_conv), dim=1)
            actor_lstm_input.append(actor_merged)

        # ACTOR
        actor_lstm_input = torch.stack(actor_lstm_input, dim=1)
        # print(actor_lstm_input.shape[2])
        actor_lstm, _ = self.actor_lstm(actor_lstm_input)

        actor_dense = self.actor_dense(actor_lstm[:, -1, :])
        actor_output = self.actor_output(actor_dense)

        mu = actor_output
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)

        return dist

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0., std=0.1)
            nn.init.constant_(m.bias, 0.1)



class Value(nn.Module):
    def __init__(self, eyes_input_shape, ears_input_shape, action_space, lstm_timestep=10, std=0.0):
        super(Value, self).__init__()
        self.lstm_timestep = lstm_timestep
        # Define hyper parameters for models (example lstm_hidden layers, conv filters ...)

        # CONVOLUTION LAYERS
        self.value_eyes_conv = nn.Sequential(
            nn.Conv2d(eyes_input_shape[0], 32, (3, 3), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1),
            nn.Conv2d(32, 64, (5, 5), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1),
            nn.Flatten(),
        )

        self.value_ears_conv = nn.Sequential(
            nn.Conv1d(ears_input_shape[0], 32, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(3, padding=1),
            nn.Conv1d(32, 64, 5, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(3, padding=1),
            nn.Flatten()
        )

        # LSTM
        # the input size should change depending on the num of features
        # ex: input of lstm = (batch, lstm_timestep, num_of_features), then input size = num_of_features
        self.value_lstm = nn.LSTM(input_size=8448, hidden_size=10, num_layers=5, batch_first=True)

        # DENSE
        self.value_dense = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # OUTPUT
        self.value_output = nn.Sequential(
            nn.Linear(64, 1)
        )

        # UTILS
        self.log_std = nn.Parameter(torch.ones(1, action_space) * std)
        self.apply(self.init_weights)

    def forward(self, observation):  # Works on single observation not batch
        value_lstm_input = []
        for observation_frame in observation:
            eye_input, ear_input = torch.FloatTensor([observation_frame[0]]), torch.FloatTensor([observation_frame[1]])
            value_eye_conv, value_ear_conv = self.value_eyes_conv(eye_input), self.value_ears_conv(ear_input)

            value_merged = torch.cat((value_eye_conv, value_ear_conv), dim=1)
            value_lstm_input.append(value_merged)

        # CRITIC
        value_lstm_input = torch.stack(value_lstm_input, dim=1)
        value_lstm, _ = self.value_lstm(value_lstm_input)

        value_dense = self.value_dense(value_lstm[:, -1, :])
        value_output = self.value_output(value_dense)

        return value_output

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0., std=0.1)
            nn.init.constant_(m.bias, 0.1)

class LaV():
    def __init__(self, eyes_input_shape, ears_input_shape, action_space, lstm_timestep=10, std=0.0):
        self.actor = Actor(eyes_input_shape, ears_input_shape, action_space, lstm_timestep=lstm_timestep, std=std)
        self.value = Value(eyes_input_shape, ears_input_shape, action_space, lstm_timestep=lstm_timestep, std=std)