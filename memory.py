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
        pass

    def clear(self):
        self.memory["eyes"] = []
        self.memory["ears"] = []
        self.memory["action"] = []
        self.memory["discounted_reward"] = []
        self.memory["advantage"] = []

    def copy(self):
        return copy.deepcopy(self.memory)
