import numpy as np
import copy
import random
import matplotlib.pyplot as plt


class memory():
    def __init__(self, gamma = 0.99):
        self.memory = {}
        self.clear()
        self.gamma = gamma
        self.lamda = 0.95
        self.clipped_reward = -10

    def store_in_memory(self, unit):
        state, action_sample, log_probs, discounted_reward, value, new_value, done = unit

        self.memory["state"].append(state)
        self.memory["action_sample"].append(action_sample)
        self.memory["log_probs"].append(log_probs)
        self.memory["discounted_reward"].append(discounted_reward)
        self.memory["value"].append(value)
        self.memory["new_value"].append(new_value)
        self.memory["done"].append(done)

    def yield_minibatches(self, minibatch_size):
        batch_size = len(self.memory["state"])
        start_index = 0
        end_index = minibatch_size

        while end_index < batch_size:
            yield self.extract_unit(start_index, end_index)
            start_index += minibatch_size
            end_index += minibatch_size

        yield self.extract_unit(start_index, batch_size)


    def extract_unit(self, start_index, end_index):
        rewards = self.memory["discounted_reward"][start_index:end_index].copy()
        values = self.memory["value"][start_index:end_index].copy()
        done = self.memory["done"][start_index:end_index].copy()
        new_values = self.memory["new_value"][start_index:end_index].copy()
        # disc_r = self.discount_rewards(rewards)
        # advantages = []
        # for i in range(len(disc_r)):
        #     advantages.append(disc_r[i] - values[i][0])
        advantages, target = self.gae(np.array(values, dtype= "float32"), np.array(new_values, dtype= "float32"), np.array(rewards, dtype= "float32"), done)

        unit = self.memory["state"][start_index:end_index],\
               self.memory["action_sample"][start_index:end_index], \
               self.memory["log_probs"][start_index:end_index], \
               advantages, \
               target, \
               values

        return self.shuffle(unit)

    def shuffle(self, unit):
        idx = [i for i in range(len(unit[0]))]
        random.shuffle(idx)

        new_unit = [[] for i in range(len(unit))]
        for i in range(len(unit[0])):
            for j in range(len(unit)):
                new_unit[j].append(unit[j][idx[i]])
        return (new_unit)


    def discount_rewards(self, reward_arr, vals_arr):
        running_add = 0
        discounted_r = np.zeros_like(reward_arr, dtype="float32")
        for i in reversed(range(0, len(reward_arr))):
            running_add = running_add * self.gamma + reward_arr[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r)  # normalizing the result
        discounted_r /= (np.std(discounted_r) + 1e-8)  # divide by standard deviation
        return discounted_r

    def gae(self, value, new_value, reward, done):
        batch_size = len(value)
        end = False

        advantage = np.zeros(batch_size + 1, dtype= "float32")
        for t in reversed(range(batch_size)):
            mask = done[t]
            if done[t]: end = True
            if reward[t] <= -99: reward[t] = self.clipped_reward
            delta = reward[t] + (self.gamma * new_value[t] * mask) - value[t]
            advantage[t] = delta + (self.gamma * self.lamda * advantage[t + 1] * mask)

        advantage = advantage[:batch_size]
        value_target = reward

        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # value_target = (value_target - value_target.mean()) / (value_target.std() + 1e-8)

        return advantage, value_target




    def clear(self):
        self.memory["state"] = []
        self.memory["action_sample"] = []
        self.memory["log_probs"] = []
        self.memory["discounted_reward"] = []
        self.memory["value"] = []
        self.memory["new_value"] = []
        self.memory["done"] = []

    def copy(self):
        return copy.deepcopy(self.memory)
