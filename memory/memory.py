import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
main_scope = os.path.join(current_dir, "..\..", "AI")
sys.path.append(main_scope)

from memory import memory_utils

_MEMORY_CHECKPOINT = 1000

class ST_MemoryUnit:
    def __init__(self, minibatch_size, gamma=0.99):
        self.memory = {}
        self.memory_being_trained = {}
        self.gamma = gamma
        self.minibatch_size=minibatch_size
        memory_utils.initialize_memory(self)
    
    def read(self):
        memory_utils.set_up_for_training(self)
        yield from memory_utils.yield_minibatches(self)
        
    def write(self, unit):
        memory_utils.store_in_memory(self, unit)
    
    def length(self):
        return len(self.memory["eyes"])
    
class LT_MemoryUnit:
    def __init__(self, minibatch_size, memory_checkpoint, gamma=0.99):
        self.memory = {}
        self.memory_being_trained = {}
        self.gamma = gamma
        self.minibatch_size=minibatch_size
        self.memory_checkpoint=memory_checkpoint
        memory_utils.initialize_memory(self)
    
    def read(self):
        memory_utils.set_up_for_training(self, long_term=True)
        yield from memory_utils.yield_minibatches(self)
        
    def write(self, unit):
        memory_utils.store_in_memory(self, unit)
        memory_utils.long_term_write(self)

    def length(self):
        return len(self.memory["eyes"])
    

class Memory:
    def __init__(self, minibatch_size, 
                 gamma=0.99, 
                 memory_checkpoint=_MEMORY_CHECKPOINT, 
                 lt_batch_size=None):
        
        if minibatch_size <= 1:
            raise ValueError(f"mini batch size should be greater than 1, minibatch provided {minibatch_size}")
        if not lt_batch_size:
            lt_batch_size=minibatch_size
        self.ST_memory = ST_MemoryUnit(minibatch_size, gamma)
        self.LT_memory = LT_MemoryUnit(lt_batch_size, memory_checkpoint, gamma)

    def read_ST(self):
        yield from self.ST_memory.read()

    def read_LT(self):
        yield from self.LT_memory.read()

    def write(self, unit):
        self.ST_memory.write(unit)
        self.LT_memory.write(unit)
    
    def length(self):
        return self.ST_memory.length()
