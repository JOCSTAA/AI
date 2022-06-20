import random
from model import LaWV

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

        # self.model_in_training = LAV(6, 2, 6).to(device)
        self.model_in_training = LaWV(eyes_input_shape, ears_input_shape, action_space, lstm_timestep=lstm_timestep).to(
            device)
        self.current_model = copy.deepcopy(self.model_in_training)
        self.optimizer = optim.Adam(self.model_in_training.parameters(), lr=learning_rate)

    # PPO LOSS FUNCTION
    def train_model(self):
        random.shuffle(self.batch_being_trained)
        for _ in range(self.epoch):
            for unit in self.batch_being_trained:
                state, action, old_log_probabilities, discounted_reward, advantage = unit
                dist, value = self.model_in_training(state)

                advantage = advantage.detach()
                old_log_probabilities = old_log_probabilities.detach()
                value = value.detach()

                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probabilities).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.loss_clip_parameter,
                                    1.0 + self.loss_clip_parameter) * advantage

                actor_loss = - torch.min(surr1, surr2).mean()
                value_loss = (discounted_reward - value).pow(2).mean()

                loss = (self.value_coefficient * value_loss) + actor_loss - (self.entropy_coefficient * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

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
        dist, value = self.current_model(observation)
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
        self.current_model = copy.deepcopy(self.model_in_training)

        self.batch_being_trained = self.batch_being_filled.copy()
        self.training = True  # Initiates next training thread

        self.batch_being_filled = []

    # TERMINATES LEARNING PROCESS
    def kill_brain(self):
        self.alive = False
        self.thread.join()