import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
main_scope = os.path.join(current_dir, "..\..", "AI")
sys.path.append(main_scope)

from model import model
import numpy as np

import unittest

class TestMemory(unittest.TestCase):

    def test_predict_valid_shape(self):
        pass

    def test_individual_model_output_shapes(self):
        pass

    def test_model_save_load_equal(self):
        pass