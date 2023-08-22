import cv2

eyes = cv2.VideoCapture(0)

for i in range(100):
    ret, image = eyes.read()
    cv2.imshow('image', image)
    cv2.waitKey(0)