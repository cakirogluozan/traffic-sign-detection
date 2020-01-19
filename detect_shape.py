from SignDetector import SignDetector
import cv2, os
from datetime import datetime


ROOT_DIR          = os.getcwd()
CAMERA_RESOLUTION = (640, 480)
CAMERA_FPS        = 20

RECORD_VIDEO      = False
RECORDER_FPS      = 5

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

if RECORD_VIDEO:

    now                            = datetime.now()
    MINUTE, HOUR, DAY, MONTH, YEAR = now.minute, now.hour, now.day, now.month, now.year
    DATE_IND                       = "_{}{}{}-{}{}".format(YEAR, MONTH, DAY, HOUR, MINUTE)

    out_video_name  = 'out{}.avi'.format(DATE_IND)
    OUT_VIDEO_DIR   = os.path.join(ROOT_DIR, total_video_name)
    OUT_VIDEOWRITER = cv2.VideoWriter(OUT_VIDEO_DIR, cv2.VideoWriter_fourcc('M','J','P','G'), RECORDER_FPS, CAMERA_RESOLUTION)


sign_detector = SignDetector('both')

while cap.isOpened():

    ret, image     = cap.read()
    detected_signs = sign_detector.detect(image, display=True)
    cv2.imshow('preprocessed_image', sign_detector.preprocessed_image)
    cv2.waitKey(1)
    
    if RECORD_VIDEO:
        out_VIDEOWRITER.write(detected_signs)

