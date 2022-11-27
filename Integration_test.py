import Input
import Output
import brain_tf
from threading import Thread
import time

lstm_cnt = 5

output = Output.Output()
input = Input.Input(lstm_cnt)

brain = brain_tf.Brain(lstm_timestep=lstm_cnt, eyes_input_shape=input.shape()[0], ears_input_shape=input.shape()[1])

# initialize
input_to_predict = input.collect_input()
action = brain.predict(input_to_predict)
output.act(action)

# do this for ever
while True:
    begin = time.time()
    # print("new training")
    # start a new thread for each new training session
    training_thread = Thread(target=brain.update_model)
    training_thread.start()
    # while training is happening in the background: act on environment
    while training_thread.is_alive():
        input_to_predict = input.collect_input()
        action = brain.predict(input_to_predict)
        output.act(action)
    # once training is done join and start new thread
    training_thread.join()
    end = time.time()
    duration = end - begin
    # print("time taken :", duration, " seconds")