import Speak
import numpy as np


from matplotlib import pyplot as plt

x = np.arange(512)
y = 200 * np.sin(2 * np.pi * np.arange(512) * 400 / 44100)

plt.plot(x,y)
plt.show()