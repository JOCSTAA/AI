import Input
import Output
import Brain
import brain_tf
import numpy as np

lstm_cnt = 5

output = Output.Output()
input = Input.Input(lstm_cnt)

brain = brain_tf.Brain(lstm_timestep=lstm_cnt, eyes_input_shape=input.shape()[0], ears_input_shape=input.shape()[1],
                       threading=False)
while True:
    input_to_predict = input.collect_input()
    action = brain.predict(input_to_predict)
    output.act(action)