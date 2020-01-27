import cv2
import os
import numpy as np
image_list = [ img for img in os.listdir() if img.endswith('png')]

for image_name in image_list:

    image = cv2.imread(image_name)
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('i', gray)
    gray = cv2.resize(gray, (64, 64))
    gray.astype(np.uint8)
    cv2.waitKey(0)
    print(image.shape)
    cv2.imwrite(image_name, gray)
