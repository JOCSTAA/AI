import numpy as np
import copy


class memory():
    def __init__(self):
        self.memory = {}
        self.clear()

    def store_in_memory(self, unit):
        eyes, ears, action, discounted_reward, advantage = unit

        self.memory["eyes"].append(eyes)
        self.memory["ears"].append(ears)
        self.memory["action"].append(advantage)
        self.memory["discounted_reward"].append(discounted_reward)
        self.memory["advantage"].append(advantage)

    def yield_minibatches(self, minibatch_size):
        batch_size = len(self.memory["eyes"])
        start_index = 0
        end_index = minibatch_size

        while end_index < batch_size:
            yield self.extract_unit(start_index, end_index)
            start_index += minibatch_size
            end_index += minibatch_size

        yield self.extract_unit(start_index, batch_size)


    def extract_unit(self, start_index, end_index):
        unit = self.memory["eyes"][start_index:end_index], \
               self.memory["ears"][start_index:end_index], \
               self.memory["action"][start_index:end_index], \
               self.memory["discounted_reward"][start_index:end_index], \
               self.memory["advantage"][start_index:end_index]
        return unit

    def clear(self):
        self.memory["eyes"] = []
        self.memory["ears"] = []
        self.memory["action"] = []
        self.memory["discounted_reward"] = []
        self.memory["advantage"] = []

    def copy(self):
        return copy.deepcopy(self.memory)
