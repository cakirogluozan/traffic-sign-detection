import cv2, imutils
import numpy as np

def shape_detection(c):
    shape = "unidentified"
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    # if the shape is a triangle, it will have 3 vertices
    if len(approx) == 3:
	    shape = "triangle"

    
    # if the shape is a pentagon, it will have 6 vertices
    elif len(approx) == 6:
	    shape = "hexagon"

    # otherwise, we assume the shape is a circle
    else:
	    shape = "circle"

    # return the name of the shape
    return shape



lower_blue = np.array([85,100,100])
upper_blue = np.array([130,255,255])

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, image = cap.read()

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue) 
    image = cv2.bitwise_and(image, image, mask = mask_blue)
    blue_masked_image = image
    ratio = 1
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)[1]

    # find contours in the thresholded image and initialize the
    # shape detector
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	    cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)

    for ind, c in enumerate(cnts):
        area = cv2.contourArea(c)
        '''
        if area < 5000:
            continue
        '''
        print(area)
        M = cv2.moments(c)
        cX = int((M["m10"] / M["m00"]) * ratio)
        cY = int((M["m01"] / M["m00"]) * ratio)
        shape = shape_detection(c)
     
        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
	        0.5, (255, 255, 255), 2)
     
        # show the output image

    cv2.imshow("Image", image)
    cv2.waitKey(1)


