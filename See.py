import cv2
import time

cam = cv2.VideoCapture(0)

while True:
    start = time.time_ns()
    result, image = cam.read()
    end = time.time_ns()

    print(end - start)
    if result:
        cv2.imshow("image", image)

    else:
        print("maaa")
