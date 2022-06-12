import cv2
import torch
import time
import numpy as np
import Hear
import Speak
from threading import Thread

eyes = cv2.VideoCapture(0)
ears = Hear.Hear()
mouth = Speak.Speak()

mouth.start_speaking()
ears.start_listening()
ret = True
total, cnt = 0, 0

while True:
    begin = time.time()

    ret, image = eyes.read()
    audio = ears.read()
    mouth.write(np.array([0.5,0.5]))

    end = time.time()


    time.sleep(0.04)#sweet spot for time to process image is about 40ms , not sure though
    duration = end-begin
    print("\n\n",image.shape, audio.shape)
    print(duration, cnt)
    FPS = 1/(duration+0.000000000000000000001)
    print("FPS is = ", FPS)
    if FPS < 10000:
        total += FPS
        cnt += 1
        avg = total/cnt
        print("AVG = ", avg)