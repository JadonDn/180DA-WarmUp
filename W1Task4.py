# Majority of the code is adopted from https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
# as well as https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
# HSV showed typically better results than using RGB
# The threshold range to capture was H: 70 units, S: 205 units, V: 235 units

# Changes in lighting showed subtle improvement to tracking ability, but not by any significant factor
# When comparing the color on the phone however, the darkest brightness setting and highest brightness setting showed major differences
# Higher brightness helped with tracking compared to lower brightness, which essentially removed the object out of tracking

import cv2 as cv
import numpy as np
 
cap = cv.VideoCapture(0)

lower_threshold = np.array([20, 50, 20])
upper_threshold = np.array([90, 255, 255])
while(1):
 
    # Take each frame
    _, frame = cap.read()
 
    # Convert BGR to HSV as it showed a better capture in variable lighting
    transcodedFrame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
 
    # Threshold the HSV image to get only green colors
    mask = cv.inRange(transcodedFrame, lower_threshold, upper_threshold)
 
    contours,_ = cv.findContours(mask, 1, 2)

    if contours:

        cnt = max(contours, key=cv.contourArea)
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(frame,[box],0,(0,0,255),2) # Red bounding box
        cv.drawContours(mask,[box],0,(255, 255, 255),2) # Construct white bounding box here in the threshold image
    # Draw frames
    cv.imshow('frame',frame)
    cv.imshow('rawThreshold',mask)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
 
cv.destroyAllWindows()