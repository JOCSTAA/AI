import cv2
import Hear

class Input():
    def __init__(self):
        self.eyes = cv2.VideoCapture(0)
        self.ears = Hear.Hear()
        self.ears.start_listening()

    def observe(self):
        ret, image = self.eyes.read()
        audio = self.ears.read()
        return [image, audio]