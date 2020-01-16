import cv2 
import numpy as np 
  
cap = cv2.VideoCapture(0)

lower_blue = np.array([85,125,100])
upper_blue = np.array([130,255,255])

while cap.isOpened():
    # Read image. 
    ret, img = cap.read()
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue) 
    img = cv2.bitwise_and(img, img, mask = mask_blue)
    blue_masked_image = img
    # Convert to grayscale. 
    gray = cv2.cvtColor(blue_masked_image, cv2.COLOR_BGR2GRAY) 
      
    # Blur using 3 * 3 kernel. 
    gray_blurred = cv2.blur(gray, (3, 3)) 
      
    # Apply Hough transform on the blurred image. 
    detected_circles = cv2.HoughCircles(gray_blurred,  
                       cv2.HOUGH_GRADIENT, 1, 300, param1 = 100, 
                   param2 = 40, minRadius = 1, maxRadius = 400) 

    # Draw circles that are detected. 
    if detected_circles is not None: 
        print(detected_circles.shape)
        # Convert the circle parameters a, b and r to integers. 
        detected_circles = np.uint16(np.around(detected_circles)) 
        
        for pt in detected_circles[0, :]: 
            a, b, r = pt[0], pt[1], pt[2] 
      
            # Draw the circumference of the circle. 
            cv2.circle(img, (a, b), r, (0, 255, 0), 2) 
      
            # Draw a small circle (of radius 1) to show the center. 
            cv2.circle(img, (a, b), 1, (0, 0, 255), 3) 
    cv2.imshow("Detected Circle", img) 
    cv2.waitKey(1) 
