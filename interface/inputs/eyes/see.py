import cv2
import numpy as np
import matplotlib.pyplot as plt


class See():
    def __init__(self,
                 image_size = (224, 224, 1),
                 normalize = True,
                 standardize = False,
                 ):
        self.width = image_size[0]
        self.height = image_size[1]

        if normalize & standardize:
            raise ValueError(f"can't use normalizaition and standardization for images")
        self.normalize = normalize
        self.standardize = standardize
        self.cam = cv2.VideoCapture(0)

    def read(self):
        _, image = self.cam.read()

        image = self.pre_process_image(image)
        
        return image
    
    def stop_seeing(self):
        self.cam.release()
        cv2.destroyAllWindows()

    def pre_process_image(self, image):
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        if self.normalize: image = image / 255.0

        if self.standardize:
            mean = np.mean(image)
            std = np.std(image)
        
            image = (image - mean) / std

        return image
    
    def show_current_image(self):
        img = self.read()
        plt.imshow(img)  # 'gray' colormap for grayscale images
        plt.axis('off')  # Turn off axis labels and ticks
        plt.show()