import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
main_scope = os.path.join(current_dir, "..\..\..", "AI")
sys.path.append(main_scope)

import numpy as np
import interface.outputs.output as Output

import unittest

class TestOutputs(unittest.TestCase):
    
    def testValidOutputs(self):
        action_space = 16
        mouth_shape = (4, 4)

        outputs = Output.Output(action_space=action_space, mouth_shape=mouth_shape)

        outputs.close_outputs()

    def testIncorrectActionSpaceSizeRaisesError(self):
        action_space = 10
        mouth_shape = (4, 4)

        with self.assertRaises(ValueError) as context:
            Output.Output(action_space=action_space, mouth_shape=mouth_shape)

        self.assertIn("given number of output elements does not match required amount,", str(context.exception)) 
