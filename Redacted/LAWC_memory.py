import numpy as np
import copy


class memory():
    def __init__(self, gamma = 0.99):
        self.memory = {}
        self.clear()
        self.gamma = gamma

    def store_in_memory(self, unit):
        state, action_sample, log_probs, discounted_reward, value, new_state = unit

        self.memory["state"].append(state)
        self.memory["action_sample"].append(action_sample)
        self.memory["log_probs"].append(log_probs)
        self.memory["reward"].append(discounted_reward)
        self.memory["value"].append(value)
        self.memory["new_state"].append(new_state)

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
        rewards = self.memory["reward"][start_index:end_index]
        values = self.memory["value"][start_index:end_index]
        disc_r = self.discount_rewards(rewards)
        advantages = []
        for i in range(len(disc_r)):
            advantages.append(disc_r[i] - values[i][0])

        unit = self.memory["state"][start_index:end_index],\
               self.memory["action_sample"][start_index:end_index], \
               self.memory["log_probs"][start_index:end_index], \
               rewards, \
               advantages, \
               self.memory["new_state"][start_index:end_index]

        return unit

    def discount_rewards(self, rewards):
        running_add = 0
        discounted_r = np.zeros_like(rewards, dtype="float32")
        for i in reversed(range(0, len(rewards))):
            running_add = running_add * self.gamma + rewards[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r)  # normalizing the result
        discounted_r /= (np.std(discounted_r) + 1e-8)  # divide by standard deviation
        return discounted_r


    def clear(self):
        self.memory["state"] = []
        self.memory["action_sample"] = []
        self.memory["log_probs"] = []
        self.memory["reward"] = []
        self.memory["value"] = []
        self.memory["new_state"] = []

    def copy(self):
        return copy.deepcopy(self.memory)
