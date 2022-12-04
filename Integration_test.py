import Input
import Output
import Brain
from threading import Thread
import time

lstm_cnt = 3
eyes_shape = (200, 200)
ears_shape = (200, 200)
outputs = Output.Output()
inputs = Input.Input(eyes_shape, ears_shape, lstm_cnt)

brain = Brain.Brain(lstm_timestep=lstm_cnt, eyes_input_shape=inputs.shape()[0], ears_input_shape=inputs.shape()[1])


# initialize
input_to_predict = inputs.collect_input()
action = brain.predict(input_to_predict)
outputs.act(action)

# do this for ever
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