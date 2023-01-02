import numpy as np
import Brain
import time

lstm_cnt = 3
eye_shape = (200, 200, 3)
ear_shape = (200, 200, 3)

brain = Brain.Brain(lstm_timestep=lstm_cnt, eyes_input_shape=eye_shape, ears_input_shape=ear_shape,
                         minibatch_size=5, batch_size=20)

random_eye_array = np.random.randint(low=0, high=255, size=eye_shape)
random_ear_array = np.random.randint(low=0, high=255, size=ear_shape)
eyes_time_sequence_observation = []
ears_time_sequence_observation = []

for _ in range(lstm_cnt):
    eyes_time_sequence_observation.append(random_eye_array)
    ears_time_sequence_observation.append(random_ear_array)

input_to_predict = (np.array(eyes_time_sequence_observation), np.array(ears_time_sequence_observation))

while True:
    begin = time.time()

    action = brain.predict(input_to_predict)
    print(action)
    brain.update_model()

    end = time.time()
    duration = end - begin
    print("time taken :", duration, " seconds")
