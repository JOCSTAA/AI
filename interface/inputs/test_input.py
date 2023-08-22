import input as input
from PIL import Image
import numpy as np

lstm_cnt = 3
eyes_shape = (200, 200)
ears_shape = (200, 200)

inputs = input.Input(eyes_shape, ears_shape, lstm_cnt)

image, audio = inputs.observe()

img = Image.fromarray(image)
img.show()

aud = Image.fromarray(audio)
aud.show()
