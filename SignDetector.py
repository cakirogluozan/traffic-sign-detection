import cv2
import numpy as np
import imutils
from model import lenet, mobilenet
from utils import get_distance, standard_deviation
from time import time
from scipy.ndimage.morphology import binary_fill_holes
from skimage import morphology
from class_dict import class_dict

class SignDetector:

    def __init__(self, mask_color):
        """
        ShapeDetector constructor

        Parameters
        ==========
        mask_color: str
            mask_color in ["blue", "red", "both"] 
        """
        # HSV FILTER PARAMETERS
        self.mask_color = mask_color

        self.lower_blue = np.array([110,109,66])
        self.upper_blue = np.array([130,255,255])
        
        self.lower_red_pos = np.array([172,166,66])
        self.upper_red_pos = np.array([180,255,255])

        self.lower_red_neg = np.array([0,166,66])
        self.upper_red_neg = np.array([8,255,255])

        self.lower_white   = np.array([0,0,200])
        self.upper_white   = np.array([180,30,255])
        
        # PREDEFINED TRAFFIC SIGN FILTERS
        '''
        self.noentry_filter     = np.array([0.368, 0.264, 0.368])
        self.pedestcross_filter = np.array([0.176, 0.705, 0.119])
        self.turnright_filter   = np.array([0.290, 0.131, 0.579])
        
        self.noentry_vconst      = np.array([12, 8, 12])
        self.pedestcross_vconst  = np.array([6, 22, 4])
        self.turnright_vconst    = np.array([9, 4, 19])
        '''
        self.cnst = 2
        self.noentry_vfilter     = np.array(self.cnst * 12*[-1] + self.cnst * 8*[1] + self.cnst * 12*[-1])
        self.pedestcross_vfilter = np.array(self.cnst * 6*[-1]  + self.cnst * 22*[1] + self.cnst * 4*[-1])
        self.turnright_vfilter   = np.array(self.cnst * 9*[-1]  + self.cnst * 4*[1] + self.cnst * 19*[-1])

        self.noentry_hfilter     = np.array(self.cnst * 16*[1])
        self.pedestcross_hfilter = np.array(self.cnst * 3*[-1] + self.cnst * 6*[1]  + self.cnst * 2*[-1] + self.cnst * 2*[1] + self.cnst * 3*[-1])
        self.turnright_hfilter   = np.array(self.cnst * 4*[1]  + self.cnst * 5*[-1] + self.cnst * 3*[1]  + self.cnst * 4*[-1])

        # GAUSSIAN FILTER PARAMETERS
        self.blurred_filter = (5, 5)
        self.blurred_param2 = 0

        # BOUNDINGBOX and NOISE CANCELLING PARAMETERS
        self.contour_area_lower_threshold   = 900
        self.contour_area_higher_threshold  = 400000
        self.contour_ratio_lower_threshold  = 5/6
        self.contour_ratio_higher_threshold = 6/5
        self.minimum_isolated_pixel_area    = 899
        self.triangle_std_threshold = 50
        self.circle_std_threshold   = 10

        # IMAGE CLASSIFICATION PARAMETERS
        self.keras_weights_dir      = 'models/lenetv9.h5'
        self.input_shape            = (64, 64, 1)
        self.include_RGB            = False
        self.num_classes            = 4
        # self.keras_model            = lenet(self.keras_weights_dir, self.input_shape, self.num_classes)
        self.class_dict             = {0: 'pedest-crossing', 1: 'no-entry', 2: 'turn-right'}
        self.pred_confidency        = 0.75

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
        mask_red_pos      = cv2.inRange(hsv, self.lower_red_pos, self.upper_red_pos)
        mask_red_neg      = cv2.inRange(hsv, self.lower_red_neg, self.upper_red_neg)
        red_masked_image  = cv2.bitwise_and(self.image, self.image, mask = mask_red_neg + mask_red_pos)

        return red_masked_image

    def whitemask_image(self):
        """
        Masking frame with predefined white filters.

        Returns
        =======
        white_masked_image: np.ndarray
        """

        hsv                = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        mask_white         = cv2.inRange(hsv, self.lower_white, self.upper_white)
        white_masked_image = cv2.bitwise_and(self.image, self.image, mask = mask_white)

        return white_masked_image

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

        binary_masked_image = self.masked_image[:,:,0] + self.masked_image[:,:,1] + self.masked_image[:,:,2]
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
        contours_obj       = cv2.findContours(contour_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_list      = imutils.grab_contours(contours_obj)
        
        return contours_obj, contours_list

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
        contours_obj, contours_list   = self.find_contours()

        for ind, contour in enumerate(contours_list):
            contour_area = cv2.contourArea(contour)
            
            if contour_area < self.contour_area_lower_threshold or contour_area > self.contour_area_higher_threshold:
                continue
            else:

                x, y, w, h     = cv2.boundingRect(contour)
                center_of_bbox = (x + w//2, y + h//2)
                if w/h > self.contour_ratio_higher_threshold or w/h < self.contour_ratio_lower_threshold:
                    continue
               
                cropped_image = self.image[y:y+h, x:x+w]
                score, class_ = self.detect_traffic_sign(cropped_image, contour, center_of_bbox)

                if score is None:
                    continue

                if display:
                    self.visualize_object(contour, (x,y,w,h), class_, score)
                    
        if display:
            cv2.imshow("Frame-with-detected-objects", self.image)
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

        if self.include_RGB:
            expanded = np.expand_dims(resized, axis=[0])
        else:
            gray     = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) 
            crosswalk_score, noentry_score, turnright_score = self.find_score(gray)
            print('crosswalk: ',crosswalk_score, 'noentry: ', noentry_score, 'turnright: ', turnright_score)
        

        preds      = np.array([crosswalk_score, noentry_score, turnright_score])/64
        max_score  = np.amax(preds)
        max_ind    = np.where(preds == max_score)[0]
        class_     = self.class_dict[max_ind[0]]
        if max_score < self.pred_confidency:
            return None, None

        std        = standard_deviation(center, contour)

        if len(max_ind) == 1:
            if max_ind in [1, 3] and std > self.circle_std_threshold or max_ind == 2 and std > self.triangle_std_threshold:
                return None, None
            else:
                print('area: {}\nstd: {}\nclass: {}\nscore: {}\nrate: {}'.format(cv2.contourArea(contour), std, class_, max_score, cropped.shape[0]/cropped.shape[1]))
                print('---'*20)
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



    def find_score(self, gray):
        bw_gray  = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
        cv2.imshow('bw', bw_gray)
        center_column = np.array(bw_gray[:,bw_gray.shape[0]//2]//255, dtype=np.int32)
        center_column[center_column==0] = -1
        half_center_row = np.array(bw_gray[bw_gray.shape[0]//2, bw_gray.shape[0]//4:3*bw_gray.shape[0]//4]//255, dtype=np.int32)
        half_center_row[half_center_row == 0] = -1
        
        
        crosswalk_score = self.find_crosswalk_matches(center_column, half_center_row)
        noentry_score   = self.find_noentry_matches(center_column, half_center_row)
        turnright_score = self.find_turnright_matches(center_column, half_center_row)
        
        return crosswalk_score, noentry_score, turnright_score

    def find_crosswalk_matches(self, center_column, half_center_row):
        crosswalk_vscore = len(np.where(center_column * self.pedestcross_vfilter== 1)[0])
        crosswalk_hscore = len(np.where(half_center_row * self.pedestcross_hfilter== 1)[0])
        crosswalk_score = sum([crosswalk_vscore/2, crosswalk_hscore])
        return crosswalk_score


    def find_noentry_matches(self, center_column, half_center_row):
        noentry_vscore = len(np.where(center_column * self.noentry_vfilter== 1)[0])
        noentry_hscore = len(np.where(half_center_row * self.noentry_hfilter== 1)[0])
        noentry_score = sum([noentry_vscore/2, noentry_hscore])
        return noentry_score

    def find_turnright_matches(self, center_column, half_center_row):
        turnright_vscore = len(np.where(center_column * self.turnright_vfilter== 1)[0])
        turnright_hscore = len(np.where(half_center_row * self.turnright_hfilter== 1)[0])
        turn_right_score = sum([turnright_vscore/2, turnright_hscore])
        return turn_right_score