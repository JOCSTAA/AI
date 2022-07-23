import numpy as np
# import Brain
import brain_tf


# TEST BRAIN AGAINST simple test model
def test_fake():
    lstm_cnt = 5
    brain = Brain.Brain(lstm_timestep=lstm_cnt)

    random_array = np.random.randint(low=0, high=255, size=(6), dtype=np.uint8)
    observation = np.array([random_array])

    initialize_param = 5

    for _ in range(initialize_param):
        action = brain.predict(observation)

        print(action)

    while True:
        if brain.training:
            action = brain.predict(observation)  # try to keep around 40ms or less

        else:
            brain.update_model()


# TEST BRAIN AGAINST simple actual model
def full_test():
    lstm_cnt = 5
    brain = Brain.Brain(lstm_timestep=lstm_cnt)

    random_eye_array = np.random.randint(low=0, high=255, size=(256, 256, 3), dtype=np.uint8)
    random_ear_array = np.random.randint(low=0, high=255, size=(20, 17), dtype=np.uint8)
    observation = (random_eye_array, random_ear_array)

    initialize_param = 5

    time_sequence_observation = []
    for _ in range(lstm_cnt):
        time_sequence_observation.append(observation)

    # time_sequence_observation = np.array([time_sequence_observation])
    for _ in range(initialize_param):
        action = brain.predict(time_sequence_observation)
        print(action)

    while True:
        print("new_loop")
        action = brain.predict(time_sequence_observation)  # try to keep around 40ms or less
        print(action)

        brain.update_model()


def full_test_tf():
    lstm_cnt = 3

    eye_shape, ear_shape = (256, 256, 3), (20, 17)
    brain = brain_tf.Brain(lstm_timestep=lstm_cnt, eyes_input_shape=eye_shape, ears_input_shape= ear_shape, threading=False)

    random_eye_array = np.random.randint(low=0, high=255, size=eye_shape)
    random_ear_array = np.random.randint(low=0, high=255, size=ear_shape)
    observation = (random_eye_array, random_ear_array)

    initialize_param = lstm_cnt

    eyes_time_sequence_observation = []
    ears_time_sequence_observation = []
    for _ in range(lstm_cnt):
        eyes_time_sequence_observation.append(observation[0])
        ears_time_sequence_observation.append(observation[1])

    input_to_predict = (np.array(eyes_time_sequence_observation), np.array(ears_time_sequence_observation))
    for _ in range(initialize_param):
        action = brain.predict(input_to_predict)
        print(action)

    # while True:
    #     if brain.training:
    #         action = brain.predict(input_to_predict)  # try to keep around 40ms or less
    #         print(action)
    #
    #     else:
    #         brain.update_model()

    ## without threading
    while True:
        # print("new_loop")
        action = brain.predict(input_to_predict)  # try to keep around 40ms or less
        print(action)

        brain.update_model()


full_test_tf()
