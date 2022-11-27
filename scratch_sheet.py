# from tesmodel import ActorW, Critic, Wonder
import numpy as np
import random

import gym

# from tesmodel import Actor, Critic

# env_name = "BipedalWalker-v3"
# env = gym.make(env_name)
#
# from tensorflow.keras import layers
# from tensorflow.keras.optimizers import Adam
#
# action_space = env.action_space.shape[0]
# input_shape = env.observation_space.shape
#
# actor = ActorW(action_space, input_shape)
# critic = Critic(input_shape)
# wonderer = Wonder(input_shape[0])
# actor.compile(optimizer=Adam(learning_rate=0.003))
# critic.compile(optimizer=Adam(learning_rate=0.003))
# wonderer.compile(optimizer=Adam(learning_rate=0.003))
#
# input_ = np.random.rand(input_shape[0])
# state = np.reshape(input_, [1, input_shape[0]])
#
# dist, step = actor(state)
#
# action_sample = dist.sample()
# action = action_sample.numpy()
#
# print(state, action)
#
# pred = wonderer((state, action))
# print(pred)

def shuffle(unit):
    idx = [i for i in range(len(unit[0]))]
    random.shuffle(idx)

    new_unit = [[] for i in range(len(unit))]
    for i in range(len(unit[0])):
        for j in range(len(unit)):
            new_unit[j].append(unit[j][idx[i]])
    return (new_unit)

unit = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2,3, 4]]
shuffle(unit)



