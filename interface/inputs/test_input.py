import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
main_scope = os.path.join(current_dir, "..\..\..", "AI")
sys.path.append(main_scope)

import numpy as np
import interface.inputs.input as Input

import unittest

class TestInputs(unittest.TestCase):

    def testIsCorrectShape(self):
        lstm_cnt = 3
        eyes_shape = (200, 200, 3)
        ears_shape = (200, 200, 1)
        inputs = Input.Input(eyes_shape, ears_shape, lstm_cnt)
        images_shape = (lstm_cnt, eyes_shape[0], eyes_shape[1], eyes_shape[2])
        audios_shape = (lstm_cnt, ears_shape[0], ears_shape[1], ears_shape[2])

        images, audios = inputs.collect_input()

        self.assertEqual(images.shape, images_shape)
        self.assertEqual(audios.shape, audios_shape)

        inputs.close_inputs()

    def testSequenceLengthisMaintained(self):
        lstm_cnt = 3
        eyes_shape = (200, 200, 3)
        ears_shape = (200, 200, 1)
        inputs = Input.Input(eyes_shape, ears_shape, lstm_cnt)
        images_shape = (lstm_cnt, eyes_shape[0], eyes_shape[1], eyes_shape[2])
        audios_shape = (lstm_cnt, ears_shape[0], ears_shape[1], ears_shape[2])

        for _ in range(10):
            images, audios = inputs.collect_input()

        self.assertEqual(images.shape, images_shape)
        self.assertEqual(audios.shape, audios_shape)

        inputs.close_inputs()

# # To view images
# import matplotlib.pyplot as plt
# lstm_cnt = 3
# eyes_shape = (200, 200, 3)
# ears_shape = (200, 200, 1)
# inputs = Input.Input(eyes_shape, ears_shape, lstm_cnt)

# images, audios = inputs.collect_input()
# image, audio = np.concatenate(images), np.concatenate(audios)

# plt.imshow(image)  # 'gray' colormap for grayscale images
# plt.axis('off')  # Turn off axis labels and ticks
# plt.show()

# plt.imshow(audio)  # 'gray' colormap for grayscale images
# plt.axis('off')  # Turn off axis labels and ticks
# plt.show()

# inputs.close_inputs()

# To calculate time for each frame || current = 0.0349
# import time

# lstm_cnt = 3
# eyes_shape = (200, 200, 3)
# ears_shape = (200, 200, 1)
# inputs = Input.Input(eyes_shape, ears_shape, lstm_cnt)
# total = 0
# samples = 300

# for i in range(samples):
#     start_time = time.time()
#     images, audios = inputs.collect_input()
#     end_time = time.time()

#     elapsed_time = end_time - start_time
#     total += elapsed_time

# avg = total/samples
# print(f'Average time: {avg:.4f}')
# inputs.close_inputs()