from interface.inputs import input as Input
import interface.outputs.output as output
import brain.brain as brain
from threading import Thread
import time

lstm_cnt = 3
eyes_shape = (200, 200, 3)
ears_shape = (200, 200, 1)
outputs = output.Output()
inputs = Input.Input(eyes_shape, ears_shape, lstm_cnt)

brain = brain.Brain(lstm_timestep=lstm_cnt,
                    eyes_input_shape=inputs.shape()[0],
                    ears_input_shape=inputs.shape()[1],
                    batch_size=2,
                    minibatch_size=2)


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
    outputs.act(action)
    print(action)
    print("training")

    brain.update_model()

    end = time.time()
    duration = end - begin
    print("time taken :", duration, " seconds")