import cv2
import numpy as np
import imutils
from model import lenet, mobilenet
from utils import distance, standard_deviation
from time import time
from scipy.ndimage.morphology import binary_fill_holes
from skimage import morphology

class SignDetector:

    def __init__(self, mask_color):
        """
        ShapeDetector constructor

        Parameters
        ==========
        mask_color: str
            mask_color in ["blue", "red", "both"] 
        """

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
        self.minimum_isolated_pixel_area    = 100

        self.keras_weights_dir      = 'models/lenetv9.h5'
        self.input_shape            = (32, 32, 1)
        self.keras_model            = lenet(self.keras_weights_dir, input_shape=self.input_shape)
        self.class_dict             = {0: 'negatives', 1: 'no-entry', 2: 'pedest-crossing', 3: 'turn-right'}
        self.pred_confidency        = 0.9
        self.triangle_std_threshold = 30
        self.circle_std_threshold   = 3


    def bluemask_image(self): 
        """
        Masking frame with predefined blue filters.

        Returns
        =======
        blue_masked_image: np.ndarray
        """

        hsv               = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        mask_blue         = cv2.inRange(hsv, self.lower_blue, self.upper_blue) 
        blue_masked_image = cv2.bitwise_and(self.image, self.image, mask = mask_blue)

        return blue_masked_image


    def redmask_image(self):
        """
        Masking frame with predefined red filters.

        Returns
        =======
        red_masked_image: np.ndarray
        """

        hsv               = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        mask_red          = cv2.inRange(hsv, self.lower_red, self.upper_red)
        red_masked_image  = cv2.bitwise_and(self.image, self.image, mask = mask_red)

        return red_masked_image


    def mask_image(self):
        """
        Masking frame according to 'mask_color'.
        
        Returns
        =======
        self.masked_image: np.ndarray
        """

        if self.mask_color   == 'red':
            self.masked_image = self.redmask_image()
        elif self.mask_color == 'blue':
            self.masked_image = self.bluemask_image()    
        elif self.mask_color == 'both':
            self.masked_image = self.redmask_image() + self.bluemask_image()    
        
        return self.masked_image

    def remove_noise(self):
        """
        Removing noise from frame by using morphology techniques.

        Returns
        =======
        noise_removed_image: np.ndarray
        """

        binary_masked_image = self.masked_image[:,:,0] + self.masked_image[:,:,2]
        holes_filled_image  = binary_fill_holes(binary_masked_image)
        noise_removed_image = np.array(morphology.remove_small_objects(holes_filled_image, min_size=self.minimum_isolated_pixel_area) * 255, dtype=np.uint8)

        return noise_removed_image


    def preprocess_image(self):
        """
        Preprocessing of raw captured frame.
        
        Returns
        =======
        self.processed_image: np.ndarray
        """

        masked_image            = self.mask_image()
        noise_removed_image     = self.remove_noise()
        self.preprocessed_image = cv2.GaussianBlur(noise_removed_image, self.blurred_filter, self.blurred_param2)
    
        return self.preprocessed_image

    def find_contours(self):
        """
        Finding contours from preprocessed image

        Returns
        =======
        contours: list
        """

        self.preprocess_image()
        contour_threshold  = cv2.threshold(self.preprocessed_image, 0, 255, cv2.THRESH_BINARY)[1]
        contours           = cv2.findContours(contour_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours           = imutils.grab_contours(contours)
        
        return contours

    def detect(self, image, display=True):
        """
        Detecting traffic signs after preprocessing image

        Parameters
        ==========
        image: np.ndarray
        display: bool

        Returns
        =======
        self.image: np.ndarray
        """

        self.image = image        
        contours   = self.find_contours()

        for ind, contour in enumerate(contours):
            contour_area = cv2.contourArea(contour)
            
            if contour_area < self.contour_area_lower_threshold or contour_area > self.contour_area_higher_threshold:
                continue
            else:
                M = cv2.moments(contour)
                cX = int((M["m10"] / M["m00"]))
                cY = int((M["m01"] / M["m00"]))

                x, y, w, h     = cv2.boundingRect(contour)
                center_of_bbox = ((2*x + w)//2, (2*y + h)//2)
                if w/h > self.contour_ratio_higher_threshold or w/h < self.contour_ratio_lower_threshold:
                    continue
               
                cropped_image = self.image[y:y+h, x:x+w]
                cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY) 
                score, class_ = self.detect_traffic_sign(cropped_image, contour, center_of_bbox)

                if score is None:
                    continue

                if display:
                    self.visualize_object(contour, (x,y,w,h), class_, score)
                    
        if display:
            "comment this through inference"
            cv2.imshow("Frame-with-detected-objects".format(self.mask_color), self.image)
        return self.image

    def detect_traffic_sign(self, cropped, contour, center):
        """
        Classifying traffic signs from cropped frame and filtering out by using standard deviation and confidency score.

        Parameters
        ==========
        cropped: np.npdarray
        contour: list
        center : list 
        
        Returns
        =======
        max_score: float
        class_   : str
        """

        resized    = cv2.resize(cropped, (self.input_shape[0], self.input_shape[1]))
        expanded   = np.expand_dims(resized, axis=[0,-1])
        preds      = np.round(self.keras_model.predict(expanded)[0], 1)

        max_score  = np.amax(preds)
        max_ind    = np.where(preds == max_score)[0]
        class_     = self.class_dict[max_ind[0]]

        if max_score < self.pred_confidency:
            return None, None

        std        = standard_deviation(center, contour)

        if len(max_ind) == 1:
            if max_ind in [0, 2] and std > self.circle_std_threshold or max_ind == 2 and std > self.triangle_std_threshold:
                return None, None
            else:
                return max_score, class_

        else:
            print('More than one maximum ind')
            return None, None

    def visualize_object(self, contour, bbox, class_, score):
        """
        Visualizing a bounding box on frame

        Parameters
        ==========
        contour: list
        bbox   : list
        class_ : str
        score  : float
        """

        x, y, w, h = bbox
        self.image = cv2.drawContours(self.image, [contour], -1, (0, 255, 0), 2)
        self.image = cv2.rectangle(self.image, (x,y), (x+w, y+h), (255,0,0), 1)
        self.image = cv2.putText(self.image, "{}:{}".format(class_, score), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


