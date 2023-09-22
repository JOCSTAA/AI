import numpy as np
import h5py
import os

_FILE_LOCATION = "memory/long_term_memory_data.h5"

def initialize_memory(memory):
    """clears the memory and passes in any data from buffer"""
    memory.memory["eyes"] = []
    memory.memory["ears"] = []
    memory.memory["action_samples"] = []
    memory.memory["log_probs"] = []
    memory.memory["rewards"] = []
    memory.memory["values"] = []
    memory.memory["new_eyes"] = []
    memory.memory["new_ears"] = []

def store_in_memory(memory, unit):
    """Store the values collected as unit into the memory"""
    eyes, ears, action_sample, log_probs, discounted_reward, value = unit

    memory.memory["eyes"].append(eyes)
    memory.memory["ears"].append(ears)
    memory.memory["action_samples"].append(action_sample)
    memory.memory["log_probs"].append(log_probs)
    memory.memory["rewards"].append(discounted_reward)
    memory.memory["values"].append(value)

def set_up_for_training(memory, long_term=False, file_location=_FILE_LOCATION):
    """
    Copies all collected data to memory being trained and then resets 
    memory being filled to start collecting more data
    """
    if long_term:
        loaded_data = {}
        with h5py.File(file_location, "r") as file:
            for key in file.keys():
                loaded_data[key] = file[key][:]

        memory.memory_being_trained = {key: np.array(val) for key, val in loaded_data.items()}

        with h5py.File(file_location, "w") as file:
            empty_dict={}
            for key, value in empty_dict.items():
                file.create_dataset(key, data=value)
    else:
        memory.memory_being_trained = {key: np.array(val) for key, val in memory.memory.items()}
    
    initialize_memory(memory)
    _relay_buffer(memory)

def yield_minibatches(memory):
    """
    Collects the values to be trained from memory and releases them
    in a step size of minibatch_size after being shuffled
    """
    _collect_wonder_data(memory)
    minibatch_size = memory.minibatch_size

    data = {key: np.array(val) for key, val in memory.memory_being_trained.items()}
    batch_size = len(data["eyes"]) - 1
    indices = np.arange(batch_size)
    np.random.shuffle(indices)

    for start_index in range(0, batch_size, minibatch_size):
        end_index = start_index + minibatch_size
        minibatch_indices = indices[start_index:end_index]

        minibatch = (
            data["eyes"][minibatch_indices],
            data["ears"][minibatch_indices],
            data["action_samples"][minibatch_indices],
            data["log_probs"][minibatch_indices],
            _collect_advantages(data["rewards"], data["values"], minibatch_indices, memory.gamma),
            data["values"][minibatch_indices],
            data["new_eyes"][minibatch_indices],
            data["new_ears"][minibatch_indices]
        )

        yield minibatch

def _collect_advantages(rewards, values, indices, gamma):
    """
    Converts the collected rewards and values to advantages, used for 
    PPO loss while maintaining their shuffled order
    """
    batched_rewards = rewards[indices]
    batched_values = values[indices]

    running_add = 0
    discounted_r = np.empty_like(batched_rewards, dtype="float32")
    for i in range(len(batched_rewards)):
        running_add = running_add * gamma + batched_rewards[len(batched_rewards) - 1 - i]
        discounted_r[len(batched_rewards) - 1 - i] = running_add

    mean_discounted_r = np.mean(discounted_r)
    std_discounted_r = np.std(discounted_r)
    normalized_discounted_r = (discounted_r - mean_discounted_r) / (std_discounted_r + 1e-8)

    advantages = normalized_discounted_r - batched_values
    return advantages

def _collect_wonder_data(memory):
    """
    Processes the memory data for training the wonder model

    Collects the t + 1 new state for every time step t
    """
    memory_data = memory.memory_being_trained
    memory_data["new_eyes"] = np.array(memory_data["eyes"][1:])
    memory_data["new_ears"] = np.array(memory_data["ears"][1:])

def _relay_buffer(memory):
    """Collect leftover data not taken in the current training batch"""
    memory_data = memory.memory_being_trained
    for key, value in memory_data.items():
        if key not in ["new_eyes", "new_ears"]:
            memory.memory[key].append(value[-1])

def long_term_write(memory, file_location=_FILE_LOCATION):
    """
    Write the saved memories to a file
    """
    if memory.length() >= memory.memory_checkpoint:
        memory.lt_length += memory.length()
        print("Saving to long term memory")

        if os.path.isfile(file_location):
            loaded_data = {}
            with h5py.File(file_location, "r") as file:
                for key in file.keys():
                    loaded_data[key] = file[key][:]

            result_dict = {}

            if loaded_data:
                for key in memory.memory.keys():
                    result_dict[key] = np.concatenate((loaded_data[key], memory.memory[key]))
            else:
                result_dict = {key: np.array(val) for key, val in memory.memory.items()}

            with h5py.File(file_location, "w") as file:
                for key, value in result_dict.items():
                    file.create_dataset(key, data=value)
        else:
            with h5py.File(file_location, "w") as file:
                for key, value in memory.memory.items():
                    file.create_dataset(key, data=value)

        initialize_memory(memory)