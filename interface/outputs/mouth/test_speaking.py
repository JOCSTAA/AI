import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
main_scope = os.path.join(current_dir, "..\..\..\..", "AI")
sys.path.append(main_scope)

import interface.outputs.mouth.speak as speak
import numpy as np

import unittest

class TestMouth(unittest.TestCase):
    
    def testValidInput(self):
        mouth = speak.Speak(mouth_shape=(512, 512))
        test_output = np.random.rand(512 * 512)

        mouth.write(test_output)
        self.assertTrue(mouth.stream.is_active())

        mouth.stop_speaking()

    def testRaisesErrorWhenOutputisNone(self):
        mouth = speak.Speak(mouth_shape=(2,2))
        test_output = None

        with self.assertRaises(ValueError) as context:
            mouth.write(test_output)

        self.assertEqual(str(context.exception), "model output can not be empty, no output provided")   

        mouth.stop_speaking()

    def testRaisesErrorWhenOutputDoesNotMatchOutputShape(self):
        mouth = speak.Speak(mouth_shape=(2,2))
        test_output = np.array([[1, 2], [3, 4]])

        with self.assertRaises(ValueError) as context:
            mouth.write(test_output)

        self.assertIn("Invalid output shape,", str(context.exception)) 

        mouth.stop_speaking()

    def testRaisesErrorWhenOutputSizeDoesNotMeetRequiredSize(self):
        mouth = speak.Speak(mouth_shape=(2,2))
        test_output = np.array([1, 2, 3, 4, 5])

        with self.assertRaises(ValueError) as context:
            mouth.write(test_output)

        self.assertIn("Invalid output size from model,", str(context.exception))

        mouth.stop_speaking() 

    def testRaisesErrorWhenAnyELementOfNNOutputIsNone(self):
        mouth = speak.Speak(mouth_shape=(2,2))
        test_output = np.array([1, 2, 3, None])

        with self.assertRaises(ValueError) as context:
            mouth.write(test_output)

        self.assertEqual(str(context.exception), "Model output contains a none element")

        mouth.stop_speaking()

# # test diagonal spectogram: increasing pitch
# spectogram = np.zeros((512, 512))
# x = 0
# y = 0
# hop = 64
# x_hop = int(len(spectogram[0])/8)
# while x < len(spectogram):
#     idx = x
#     for i in range(hop):
#         idx += i
#         if idx > len(spectogram) : break
#         spectogram[idx][y:y+x_hop - 1] = np.ones(x_hop -1)
    
#     x += hop
#     y += x_hop

# spectogram = np.array(spectogram)
# mouth = speak.Speak(mouth_shape=(512, 512))

# mouth.speak_spectogram(spectogram)

# while mouth.stream.is_active():
#     if mouth.waveform_index > len(mouth.waveform): break

# mouth.stop_speaking
