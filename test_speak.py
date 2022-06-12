import Speak
import numpy as np
import time

mouth = Speak.Speak()

mouth.start_speaking()
x = 0
timer = time.time()
while True:
    mouth.write(np.array([x, 1, 0.12, 0.5]))
    if time.time() - timer > 2:
        timer = time.time()
        x += 0.1
        if x > 1: x = x-1
