import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
main_scope = os.path.join(current_dir, "..\..\..\..", "AI")
sys.path.append(main_scope)

import numpy as np
import interface.inputs.eyes.see as See

import unittest

class TestEars(unittest.TestCase):

    def testIsCorrectShape(self):
        eyes = See.See()
        image = eyes.read()
        shape = (224, 224, 3)

        self.assertEqual(image.shape, shape)

        eyes.stop_seeing()

    def testIsRGB(self):
        eyes = See.See()
        image = eyes.read()

        self.assertEqual(len(image.shape), 3)
        self.assertEqual(image.shape[2], 3)

        eyes.stop_seeing()

    def testIsNormalized(self):
        eyes = See.See()
        image = eyes.read()

        self.assertGreaterEqual(1, np.max(image))
        self.assertGreaterEqual(np.min(image), 0)

        eyes.stop_seeing()

    def testIsStandardized(self):
        eyes = See.See(normalize=False, standardize=True)
        image = eyes.read()

        mean = np.mean(image)
        std = np.std(image)
        self.assertTrue(np.isclose(mean, 0) and np.isclose(std, 1))

        eyes.stop_seeing()

## To view images
# eyes = See.See()

# for i in range(3):
#     eyes.show_current_image()
#     print(eyes.read().shape)

# eyes.stop_seeing()

## To calculate time for each frame || current = 0.0349
# import time

# eyes = See.See()
# total = 0
# samples = 300

# for i in range(samples):
#     start_time = time.time()
#     eyes.read()
#     end_time = time.time()

#     elapsed_time = end_time - start_time
#     total += elapsed_time

# avg = total/samples
# print(f'Average time: {avg:.4f}')
# eyes.stop_seeing()