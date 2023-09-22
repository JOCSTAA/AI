import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
main_scope = os.path.join(current_dir, "..\..", "AI")
sys.path.append(main_scope)

from memory import memory
import numpy as np

import unittest

class TestMemory(unittest.TestCase):

    def test_read_write_ST(self):
        from memory import memory_utils 
        eyes = np.random.rand(3, 4, 4, 3)
        ears = np.random.rand(3, 4, 4, 1)
        new_eyes = np.random.rand(3, 4, 4, 3)
        new_ears = np.random.rand(3, 4, 4, 1)
        action_sample = np.random.rand(3, 16384)
        log_probs = np.random.rand(3,)
        reward = 1
        value = 1
        indices = np.arange(1) # batch_size - 1
        advantage = memory_utils._collect_advantages(np.array([reward, reward]), np.array([value, value]), indices, gamma=0.99)

        unit1=[eyes, ears, action_sample, log_probs, reward, value]
        unit2=[new_eyes, new_ears, action_sample, log_probs, reward, value]

        sample_memory = memory.Memory(minibatch_size=2)

        sample_memory.write(unit1)
        sample_memory.write(unit2)

        units = [unit for unit in sample_memory.read_ST()]
        read_eyes, read_ears, read_action_samples, read_old_log_probs, read_advantages, read_old_values, read_new_eyes, read_new_ears = units[0]

        self.assertEqual(len(units), 1)
        self.assertTrue(np.all(eyes == read_eyes))
        self.assertTrue(np.all(ears == read_ears))
        self.assertTrue(np.all(new_eyes == read_new_eyes))
        self.assertTrue(np.all(new_ears == read_new_ears))
        self.assertTrue(np.all(action_sample == read_action_samples))
        self.assertTrue(np.all(log_probs == read_old_log_probs))
        self.assertTrue(np.all(value == read_old_values))
        self.assertTrue(np.all(advantage == read_advantages))

    def test_invalid_minibatch_size(self):
        minibatch_size = 1

        with self.assertRaises(ValueError) as context:
            sample_memory = memory.Memory(minibatch_size=minibatch_size)

        self.assertIn("mini batch size should be greater than 1,", str(context.exception)) 

    def test_read_write_LT(self):
        import math
        memory_checkpoint = 10
        eyes = np.random.rand(3, 4, 4, 3)
        ears = np.random.rand(3, 4, 4, 1)
        new_eyes = eyes
        new_ears = ears
        action_sample = np.random.rand(3, 16384)
        log_probs = np.random.rand(3,)
        reward = 1
        value = 1
        minibatch_size = 2
        num_of_units=20
        num_of_batched_units = math.floor(num_of_units/minibatch_size)

        written_unit=[eyes, ears, action_sample, log_probs, reward, value]

        sample_memory = memory.Memory(minibatch_size=minibatch_size, 
                                      memory_checkpoint=memory_checkpoint)

        for _ in range(num_of_units):
            sample_memory.write(written_unit)

        units = [unit for unit in sample_memory.read_LT()]
        read_eyes, read_ears, read_action_samples, read_old_log_probs, _, read_old_values, read_new_eyes, read_new_ears = units[0]

        self.assertEqual(len(units), num_of_batched_units)
        self.assertTrue(np.all(eyes == read_eyes))
        self.assertTrue(np.all(ears == read_ears))
        self.assertTrue(np.all(new_eyes == read_new_eyes))
        self.assertTrue(np.all(new_ears == read_new_ears))
        self.assertTrue(np.all(action_sample == read_action_samples))
        self.assertTrue(np.all(log_probs == read_old_log_probs))
        self.assertTrue(np.all(value == read_old_values))     

    def test_memory_length(self):
        eyes = np.random.rand(3, 4, 4, 3)
        ears = np.random.rand(3, 4, 4, 1)
        action_sample = np.random.rand(3, 16384)
        log_probs = np.random.rand(3,)
        reward = 1
        value = 1
        minibatch_size = 2
        num_of_units=20

        written_unit=[eyes, ears, action_sample, log_probs, reward, value]

        sample_memory = memory.Memory(minibatch_size=minibatch_size,)

        for _ in range(num_of_units):
            sample_memory.write(written_unit)

        self.assertEqual(sample_memory.length(), num_of_units)
        
