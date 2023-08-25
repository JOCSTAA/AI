import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
main_scope = os.path.join(current_dir, "..\..\..\..", "AI")
sys.path.append(main_scope)

import numpy as np
import interface.inputs.ears.hear as Hear

import unittest

class TestEars(unittest.TestCase):

    def testIsCorrectShape(self):
        ear = Hear.Hear()
        audio = ear.read()
        shape = (224, 224, 1)

        self.assertEqual(audio.shape, shape)

        ear.stop_listening()

    def testIsGrayscale(self):
        ear = Hear.Hear()
        audio = ear.read()

        self.assertEqual(len(audio.shape), 3)
        self.assertEqual(audio.shape[2], 1)

        ear.stop_listening()
    
    def testIsNormalized(self):
        ear = Hear.Hear()
        audio = ear.read()

        self.assertGreaterEqual(1, np.max(audio))
        self.assertGreaterEqual(np.min(audio), 0)

        ear.stop_listening()

    def testIsStandardized(self):
        ear = Hear.Hear(normalize=False, standardize=True)
        audio = ear.read()

        mean = np.mean(audio)
        std = np.std(audio)
        self.assertTrue(np.isclose(mean, 0) and np.isclose(std, 1))

        ear.stop_listening()

# # To view MFCC images
# ear = Hear.Hear()

# for i in range(5):
#     print(ear.read().shape)
#     ear.show_current_MFCC()

# ear.stop_listening()

# # To calculate time for each frame || current = 0.0689
# import time

# total = 0
# samples = 300

# ear = Hear.Hear()
# ear.read()
# for i in range(samples):
#     start_time = time.time()
#     ear.read()
#     end_time = time.time()
    
#     elapsed_time = end_time - start_time
#     total += elapsed_time


# ear.stop_listening()

# avg = total/samples
# print(f'Average time: {avg:.4f}')