import numpy as np
import copy
import random
import matplotlib.pyplot as plt


def shuffle(unit):
    print(len(unit[6]))
    idx = [i for i in range(len(unit[0]))]
    random.shuffle(idx)

    new_unit = [[] for _ in range(len(unit))]
    for i in range(len(unit[0])):
        for j in range(len(unit)):
            new_unit[j].append(unit[j][idx[i]])
    return new_unit


class Memory:
    def __init__(self, gamma=0.99):
        self.memory = {}
        self.buffer = {}
        self.gamma = gamma
        self.lamda = 0.95
        self.clipped_reward = -10
        self.clear()

    def store_in_memory(self, unit):
        """Store the values collected in brain.Predict into the memory"""
        eyes, ears, action_sample, log_probs, discounted_reward, value = unit

        self.memory["eyes"].append(eyes)
        self.memory["ears"].append(ears)
        self.memory["action_sample"].append(action_sample)
        self.memory["log_probs"].append(log_probs)
        self.memory["discounted_reward"].append(discounted_reward)
        self.memory["value"].append(value)

    def yield_minibatches(self, minibatch_size):
        """Collects the values to be trained from memory and releases them in a step size of minibatch_size"""
        batch_size = len(self.memory["eyes"]) - 1
        start_index = 0
        end_index = minibatch_size

        self.collect_wonder_data()

        while end_index < batch_size:
            yield self.extract_unit(start_index, end_index)
            start_index += minibatch_size
            end_index += minibatch_size

        yield self.extract_unit(start_index, batch_size)

    def extract_unit(self, start_index, end_index):
        rewards = self.memory["discounted_reward"][start_index:end_index].copy()
        values = self.memory["value"][start_index:end_index].copy()
        disc_r = self.discount_rewards(rewards)
        advantages = np.subtract(disc_r, values)

        unit = self.memory["eyes"][start_index:end_index], \
            self.memory["ears"][start_index:end_index], \
            self.memory["action_sample"][start_index:end_index], \
            self.memory["log_probs"][start_index:end_index], \
            advantages, \
            values, self.memory["new_eyes"][start_index:end_index], self.memory["new_ears"][start_index:end_index]

        return shuffle(unit)

    def collect_wonder_data(self):
        """
        Processes the memory data for training the wonder model

        Collects the t + 1 new state for every time step t
        """
        self.collect_buffer()
        index = 1
        while index < len(self.memory["eyes"]):
            self.memory["new_eyes"].append(self.memory["eyes"][index])
            self.memory["new_ears"].append(self.memory["ears"][index])

            index += 1

    def collect_buffer(self):
        """collect left over data not taken in current training batch"""
        self.buffer = {}
        for key, value in self.memory.items():
            if key not in "new_eyes, new_ears":
                self.buffer[key] = value[-1]

    def pass_buffer(self):
        """add the collected buffer data to the newly refreshed memory"""
        for key, value in self.buffer.items():
            self.memory[key].append(self.buffer[key])

    def discount_rewards(self, reward_arr):
        running_add = 0
        discounted_r = np.zeros_like(reward_arr, dtype="float32")
        for i in reversed(range(0, len(reward_arr))):
            running_add = running_add * self.gamma + reward_arr[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r)  # normalizing the result
        discounted_r /= (np.std(discounted_r) + 1e-8)  # divide by standard deviation
        return discounted_r

    def clear(self):
        """clears the memory and passes in any data from buffer"""
        self.memory["eyes"] = []
        self.memory["ears"] = []
        self.memory["action_sample"] = []
        self.memory["log_probs"] = []
        self.memory["discounted_reward"] = []
        self.memory["value"] = []
        self.memory["new_eyes"] = []
        self.memory["new_ears"] = []

        self.pass_buffer()

    def copy(self):
        return copy.deepcopy(self.memory)
