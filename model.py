import torch
import torch.nn as nn

from torch.distributions import Normal

class LaWV(nn.Module):
    def __init__(self, eyes_input_shape, ears_input_shape, action_space, lstm_timestep=10, std=0.0):
        super(LaWV, self).__init__()
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
        self.actor_lstm = nn.LSTM(input_size=8448, hidden_size=10, num_layers=5, batch_first=True)

        self.value_lstm = nn.LSTM(input_size=8448, hidden_size=10, num_layers=5, batch_first=True)

        # DENSE
        self.actor_dense = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.value_dense = nn.Sequential(
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
        self.value_output = nn.Sequential(
            nn.Linear(64, 1)
        )

        # UTILS
        self.log_std = nn.Parameter(torch.ones(1, action_space) * std)
        self.apply(self.init_weights)

    def forward(self, observation):  # Works on single observation not batch
        actor_lstm_input = []
        value_lstm_input = []
        for observation_frame in observation:
            eye_input, ear_input = torch.FloatTensor([observation_frame[0]]), torch.FloatTensor([observation_frame[1]])
            actor_eye_conv, actor_ear_conv = self.actor_eyes_conv(eye_input), self.actor_ears_conv(ear_input)
            value_eye_conv, value_ear_conv = self.value_eyes_conv(eye_input), self.value_ears_conv(ear_input)

            actor_merged = torch.cat((actor_eye_conv, actor_ear_conv), dim=1)
            actor_lstm_input.append(actor_merged)

            value_merged = torch.cat((value_eye_conv, value_ear_conv), dim=1)
            value_lstm_input.append(value_merged)

        # ACTOR
        actor_lstm_input = torch.stack(actor_lstm_input, dim=1)
        # print(actor_lstm_input.shape[2])
        actor_lstm, _ = self.actor_lstm(actor_lstm_input)

        actor_dense = self.actor_dense(actor_lstm[:, -1, :])
        actor_output = self.actor_output(actor_dense)

        # CRITIC
        value_lstm_input = torch.stack(value_lstm_input, dim=1)
        value_lstm, _ = self.value_lstm(value_lstm_input)

        value_dense = self.value_dense(value_lstm[:, -1, :])
        value_output = self.value_output(value_dense)

        mu = actor_output
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)

        return dist, value_output

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0., std=0.1)
            nn.init.constant_(m.bias, 0.1)
