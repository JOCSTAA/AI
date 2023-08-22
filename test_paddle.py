import numpy as np
import cv2
import pyautogui
from test_environments.paddle.paddle import Paddle
from brain.brain import Brain
import time

env = Paddle()

lstm_cnt = 3
xbound = env.width - 23
ybound = env.height - 23
eye_shape = (ybound, xbound, 3)
ear_shape = (1, 1, 3)

brain = Brain(lstm_timestep=lstm_cnt, eyes_input_shape=eye_shape, ears_input_shape=ear_shape,
                    minibatch_size=64, batch_size=64, action_space=1, epoch = 5, learning_rate=0.01)


def get_image():
    image = pyautogui.screenshot(region=(10, 33, xbound, ybound))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.imwrite("image1.png", image)
    return image


random_ear_array = np.zeros(ear_shape)

eyes_time_sequence_observation = []
ears_time_sequence_observation = []


def collect_input(image, audio):
    if not eyes_time_sequence_observation:
        for _ in range(lstm_cnt):
            eyes_time_sequence_observation.append(image)
            ears_time_sequence_observation.append(audio)

    else:
        eyes_time_sequence_observation.pop(0)
        ears_time_sequence_observation.pop(0)
        eyes_time_sequence_observation.append(image)
        ears_time_sequence_observation.append(audio)

    input_to_predict = (
        np.array(eyes_time_sequence_observation), np.array(ears_time_sequence_observation))
    return input_to_predict

cnt = 0
while True:
    action = brain.predict_test((collect_input(get_image(), random_ear_array)), env)[0]
    brain.update_model()
    if cnt % 100 == 0:
        print("Count : {}  :, Hit: {}, Miss: {}".format(cnt, env.hit, env.miss))

    cnt += 1