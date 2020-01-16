import cv2
import numpy as np
import imutils
from model import lenet, mobilenet
from time import time
from scipy.ndimage.morphology import binary_fill_holes
from skimage import morphology

class Shape:

    def __init__(self, mask_color):

        self.mask_color = mask_color
        
        self.lower_blue = np.array([113,109,88])
        self.upper_blue = np.array([120,255,255])
        
        self.lower_red = np.array([168,122,66])
        self.upper_red = np.array([180,255,255])
        '''
        self.lower_blue = np.array([104,77,88])
        self.upper_blue = np.array([121,255,255])
        
        self.lower_red = np.array([160,77,99])
        self.upper_red = np.array([180,255,255])

        '''

        self.blurred_filter = (5, 5)
        self.blurred_param2 = 0

        self.contour_area_lower_threshold   = 900
        self.contour_area_higher_threshold  = 400000
        self.contour_ratio_lower_threshold  = 3/4
        self.contour_ratio_higher_threshold = 4/3

        self.keras_weights_dir  = 'models/lenetv9.h5'
        self.input_shape        = (32, 32, 1)
        self.keras_model        = lenet(self.keras_weights_dir, input_shape=self.input_shape)
        self.class_dict         = {0: 'negatives', 1: 'no-entry', 2: 'pedest-crossing', 3: 'turn-right'}

    def bluemask_image(self): 
        
        hsv                 = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        mask_blue           = cv2.inRange(hsv, self.lower_blue, self.upper_blue) 
        blue_masked_image   = cv2.bitwise_and(self.image, self.image, mask = mask_blue)
#        cv2.imshow('blue', blue_masked_image)
        return blue_masked_image

    def redmask_image(self):

        hsv                 = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        mask_red           = cv2.inRange(hsv, self.lower_red, self.upper_red)
        red_masked_image    = cv2.bitwise_and(self.image,self.image,mask = mask_red)
        red_masked_image    = cv2.bitwise_and(self.image,self.image,mask = mask_red)
        return red_masked_image

    def preprocess_image(self):

        if self.mask_color == 'red':
            masked_image = self.redmask_image()
        elif self.mask_color == 'blue':
            masked_image = self.bluemask_image()    
        elif self.mask_color == 'both':
            masked_image = self.redmask_image() + self.bluemask_image()
        masked_image_ = masked_image[:,:,0] + masked_image[:,:,2]
        color_gray                = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY) 
        mask_holes_filled = binary_fill_holes(masked_image_)
        mask_removed_noise = np.array(morphology.remove_small_objects(mask_holes_filled, min_size=400)*255, dtype=np.uint8)
        
        self.color_blurred        = cv2.GaussianBlur(mask_removed_noise, self.blurred_filter, self.blurred_param2)
        cv2.imshow('mmmm', mask_removed_noise)
        return self.color_blurred

    def find_contours(self):
        self.preprocess_image()
        self.contour_threshold = cv2.threshold(self.color_blurred, 0, 255, cv2.THRESH_BINARY)[1]
        contours               = cv2.findContours(self.contour_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contours          = imutils.grab_contours(contours)

        return contours

    def detect(self, image, visualize=True, display=True):
        self.image = image
        
        self.find_contours()
        
        x, y, w, h = None, None, None, None 
        detected_traffic_signs = list()

        for ind, contour in enumerate(self.contours):
            contour_area = cv2.contourArea(contour)
            if contour_area < self.contour_area_lower_threshold or contour_area > self.contour_area_higher_threshold:
                continue
            else:
                # print(contour_area)
                M = cv2.moments(contour)
                cX = int((M["m10"] / M["m00"]))
                cY = int((M["m01"] / M["m00"]))

                x, y, w, h = cv2.boundingRect(contour)
                bbox = (x, y, w, h)

                if w/h > self.contour_ratio_higher_threshold or w/h < self.contour_ratio_lower_threshold:
                    continue
               
                cropped_image = self.image[y:y+h, x:x+w]
                cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY) 
                score, class_ =  self.detect_traffic_sign(cropped_image, contour, bbox)

                if score is None:
                    continue

                if visualize:
                    self.image = cv2.drawContours(self.image, [contour], -1, (0, 255, 0), 2)
                    self.image = cv2.rectangle(self.image, (x,y), (x+w, y+h), (255,0,0), 1)
                    self.image = cv2.putText(self.image, "{}:{}".format(class_, score), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
        if display:
            "comment this through inference"
            cv2.imshow("Image {}".format(self.mask_color), self.image)
        return self.image
                

    

    def detect_traffic_sign(self, cropped, contour, bbox):
        
        x, y, w, h = bbox
        center     = ((2*x + w)//2, (2*y + h)//2)
        resized    = cv2.resize(cropped, (self.input_shape[0], self.input_shape[1]))
        expanded   = np.expand_dims(resized, axis=-1)
        expanded   = np.expand_dims(expanded, axis=0)
        probs      = np.round(self.keras_model.predict(expanded)[0], 1)
        max_value  = np.amax(probs)


        if max_value < 0.9:
            return None, None
        
        max_ind  = np.where(probs == max_value)[0]

        std = standard_deviation(center, contour)
        print(std)

        if len(max_ind) == 1:
            if max_ind == 0 or max_ind != 2 and std > 4 or max_ind ==2 and std > 30:
                return None, None
            else:
                return max_value, self.class_dict[max_ind[0]]

        else:
            print('More than one maximum ind')
            return None, None



def distance(x1, y1, x2, y2):
    return ((y2-y1)**2 + (x2-x1)**2)**(1/2)

def standard_deviation(center, contour):
    center_x, center_y = center
    distance_list = list()
    for point in contour:
        x = point[0][0]
        y = point[0][1]
        dist = distance(x, y, center_x, center_y)
        distance_list.append(dist)
    distance_std = np.std(np.array(distance_list))
    return distance_std

def get_center(contour):
    center = np.mean(contour, axis=0)[0]
    return center
