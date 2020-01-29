import cv2
import numpy as np
import imutils
from utils import get_distance, standard_deviation
from time import time
from scipy.ndimage.morphology import binary_fill_holes
from skimage import morphology
from class_dict import class_dict
from prettytable import PrettyTable

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
        
        self.noentry_vfilter     = np.concatenate((-np.ones(self.cnst*12), np.ones(self.cnst*8),  -np.ones(self.cnst*12)))
        self.pedestcross_vfilter = np.concatenate((-np.ones(self.cnst*6),  np.ones(self.cnst*22), -np.ones(self.cnst*4)))
        self.turnright_vfilter   = np.concatenate((-np.ones(self.cnst*9),  np.ones(self.cnst*4), -np.ones(self.cnst*19)))
        self.vfilter_size        = len(self.noentry_vfilter)

        self.noentry_hfilter     = np.ones(self.cnst*16)
        self.pedestcross_hfilter = np.concatenate((-np.ones(self.cnst*3),  np.ones(self.cnst*6), -np.ones(self.cnst*2), np.ones(self.cnst*2), -np.ones(self.cnst*3)))
        self.turnright_hfilter   = np.concatenate((np.ones(self.cnst*4), -np.ones(self.cnst*5), np.ones(self.cnst*3), -np.ones(self.cnst*4)))
        self.hfilter_size        = len(self.noentry_hfilter)

        self.error_threshold     = 0.5
        self.verbose             = True
        # GAUSSIAN FILTER PARAMETERS
        self.blurred_filter = (5, 5)
        self.blurred_param2 = 0

        # BOUNDINGBOX and NOISE CANCELLING PARAMETERS
        self.contour_area_lower_threshold   = 900
        self.contour_area_higher_threshold  = 400000
        self.contour_ratio_lower_threshold  = 5/6
        self.contour_ratio_higher_threshold = 6/5
        self.minimum_isolated_pixel_area    = 899
        self.binary_image_lower_threshold   = 150
        self.triangle_std_threshold = 50
        self.circle_std_threshold   = 10

        # IMAGE CLASSIFICATION PARAMETERS
        self.input_width            = self.cnst * 32
        self.input_height           = self.cnst * 32
        self.input_shape            = (self.input_width, self.input_height)
        self.include_RGB            = False
        self.class_dict             = {0: 'no-entry', 1: 'pedest-crossing', 2: 'turn-right'}
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

        resized    = cv2.resize(cropped, (self.input_height, self.input_width))
        gray      = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) 
        scores    = self.find_score(gray)
        wh_rate   = np.round(cropped.shape[0]/cropped.shape[1],2)
        noentry_score, pedestcrossing_score, turnright_score = scores
        
        preds      = np.array([noentry_score, pedestcrossing_score, turnright_score])
        max_score  = np.amax(preds)
        max_ind    = np.where(preds == max_score)[0]
        class_     = self.class_dict[max_ind[0]]

        if max_score < self.pred_confidency:
            return None, None

        std = standard_deviation(center, contour)

        if len(max_ind) == 1:
            if max_ind in [0, 2] and std > self.circle_std_threshold or max_ind == 1 and std > self.triangle_std_threshold:
                return None, None
            else:
                if self.verbose:
                    self.reporter(class_, max_score, contour, std, wh_rate, noentry_score, pedestcrossing_score, turnright_score)
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
        gray_to_binary  = cv2.threshold(gray, self.binary_image_lower_threshold, 255, cv2.THRESH_BINARY)[1]
        center_column   = np.array(gray_to_binary[:,self.input_width//2]//255, dtype=np.int32)
        half_center_row = np.array(gray_to_binary[self.input_height//2, self.input_width//4 : 3 * self.input_width//4]//255, dtype=np.int32)
        
        center_column[center_column == 0]     = -1
        half_center_row[half_center_row == 0] = -1
        
        cv2.imshow('bw', gray_to_binary)
        
        pedestcrossing_score = self.find_pedestcrossing_matches(center_column, half_center_row)
        noentry_score        = self.find_noentry_matches(center_column, half_center_row)
        turnright_score      = self.find_turnright_matches(center_column, half_center_row)
        scores               = noentry_score, pedestcrossing_score, turnright_score
        return scores

    def find_pedestcrossing_matches(self, center_column, half_center_row):
        pedestcrossing_vscore = len(np.where(center_column   * self.pedestcross_vfilter== 1)[0])
        pedestcrossing_hscore = len(np.where(half_center_row * self.pedestcross_hfilter== 1)[0])
        if pedestcrossing_vscore < self.error_threshold:
            pedestcrossing_vscore = 0
        if pedestcrossing_hscore < self.error_threshold:
            pedestcrossing_hscore = 0
        length_matches   = (len(center_column), len(half_center_row))
        pedestcrossing_score  = self.weight_matches(pedestcrossing_vscore, pedestcrossing_hscore, length_matches)
        return pedestcrossing_score


    def find_noentry_matches(self, center_column, half_center_row):
        noentry_vscore = len(np.where(center_column   * self.noentry_vfilter== 1)[0])
        noentry_hscore = len(np.where(half_center_row * self.noentry_hfilter== 1)[0])
        if noentry_vscore < self.error_threshold:
            noentry_vscore = 0
        if noentry_hscore < self.error_threshold:
            noentry_hscore = 0
        length_matches   = (len(center_column), len(half_center_row))
        noentry_score  = self.weight_matches(noentry_vscore, noentry_hscore, length_matches)
        return noentry_score

    def find_turnright_matches(self, center_column, half_center_row):
        turnright_vscore = len(np.where(center_column   * self.turnright_vfilter== 1)[0])
        turnright_hscore = len(np.where(half_center_row * self.turnright_hfilter== 1)[0])

        length_matches   = (len(center_column), len(half_center_row))
        turnright_score  = self.weight_matches(turnright_vscore, turnright_hscore, length_matches)
        return turnright_score

    def weight_matches(self, vscore, hscore, length_matches, punish=True, round=True, weighting_mode=None):
        vscore_len, hscore_len = length_matches
        if punish:
        
            if vscore < self.error_threshold*vscore_len:
                vscore = 0
            if hscore < self.error_threshold:
                hscore = 0

        if not weighting_mode:
            weighted_score = (hscore + vscore) / sum(length_matches)
                
        
        if round:
            weighted_score = np.round(weighted_score, 2)
        
        return weighted_score

    def reporter(self, class_, max_score, contour, std, wh_rate, noentry_score, pedestcrossing_score, turnright_score):
        output_table = PrettyTable(['Output', 'Value'])
        output_table.add_row(['class', class_])
        output_table.add_row(['max_score', max_score])
        
        stats_table = PrettyTable(['Stat', 'Value'])
        stats_table.add_row(['area', cv2.contourArea(contour)])
        stats_table.add_row(['std', std])
        stats_table.add_row(['wh_rate', wh_rate])

        preds_table = PrettyTable(['Class', 'Value'])
        preds_table.add_row(['no-entry', noentry_score])
        preds_table.add_row(['pedest-cross', pedestcrossing_score])
        preds_table.add_row(['turn-right', turnright_score])

        print(output_table)
        print(stats_table)
        print(preds_table)
        print('~~'*20)

        