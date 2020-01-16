from ShapeDetector import Shape
import cv2
from datetime import datetime

SAVE_VIDEO = False


if SAVE_VIDEO:
    now = datetime.now()
    MINUTE, HOUR, DAY, MONTH, YEAR = now.minute, now.hour, now.day, now.month, now.year
    DATE_IND = "_{}{}{}-{}{}".format(YEAR, MONTH, DAY, HOUR, MINUTE)

    red_VIDEO_DIR = 'redvideo{}.avi'.format(DATE_IND)
    red_VIDEOWRITER = cv2.VideoWriter(red_VIDEO_DIR, cv2.VideoWriter_fourcc('M','J','P','G'), 5, (320,240))

    blue_VIDEO_DIR = 'bluevideo{}.avi'.format(DATE_IND)
    blue_VIDEOWRITER = cv2.VideoWriter(blue_VIDEO_DIR, cv2.VideoWriter_fourcc('M','J','P','G'), 5, (320,240))

    total_VIDEO_DIR = 'totalvideo{}.avi'.format(DATE_IND)
    total_VIDEOWRITER = cv2.VideoWriter(total_VIDEO_DIR, cv2.VideoWriter_fourcc('M','J','P','G'), 5, (320,240))


cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
cap.set(cv2.CAP_PROP_FPS,20)


blue_shape_detector = Shape('blue')
red_shape_detector  = Shape('red')
both_shape_detector = Shape('both')

while cap.isOpened():
    ret, image = cap.read()

    # detected_blue_signs = blue_shape_detector.detect(image, display=True)
    # cv2.imshow('detected_blue_signs', detected_blue_signs)
    # detected_red_signs = red_shape_detector.detect(image, display=True)
    # cv2.imshow('detected_red_signs', detected_red_signs)
    # cv2.waitKey(1)

    detected_br_signs = both_shape_detector.detect(image, visualize=True, display=True)
    cv2.waitKey(1)
    
if SAVE_VIDEO:
        red_VIDEOWRITER.write(detected_red_signs)
        blue_VIDEOWRITER.write(detected_blue_signs)
        total_VIDEOWRITER.write(detected_blue_signs)
        total_VIDEOWRITER.write(detected_red_signs)

