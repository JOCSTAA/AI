from interface.inputs import input as Input
import interface.outputs.output as output
import brain.brain as brain
from threading import Thread
import time
import math
import numpy as np

sequence_length = 3
eyes_shape = (4, 4, 3)
ears_shape = (4, 4, 1)
shape = 128
mouth_shape = (shape, shape) 
action_space = shape * shape # breaks at > 4096 # factor of mouth_shape
outputs = output.Output(action_space, mouth_shape=mouth_shape)
inputs = Input.Input(eyes_shape, ears_shape, sequence_length)

brain = brain.Brain(sequence_length=sequence_length,
                    eyes_input_shape=inputs.shape()[0],
                    ears_input_shape=inputs.shape()[1],
                    action_space=action_space,
                    batch_size=2,
                    minibatch_size=2,
                    wonder=True)


# initialize
input_to_predict = inputs.collect_input()
action = brain.predict(input_to_predict)
outputs.act(action)

# do this forever
while True:
    begin = time.time()
    # print("new training")
    # start a new thread for each new training session
    input_to_predict = inputs.collect_input()
    action = brain.predict(input_to_predict)
    # print(action)
    # print(action.shape)
    outputs.act(action)
    print("training")

    brain.update_model()

    end = time.time()
    duration = end - begin
    print("time taken :", duration, " seconds")