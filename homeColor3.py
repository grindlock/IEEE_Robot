import cv2
import numpy as np
import argparse
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt
import rospy
from std_msgs.msg import String
from collections import deque
from imutils.video import VideoStream
import imutils
import time
import rospy
from std_msgs.msg import String
import math

def countPixels(mask):
	return cv2.countNonZero(mask)
def contourRadius(mask):
	radius = 0
	cntsColor = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
	cntsColor = imutils.grab_contours(cntsColor)#fixes according to opencv version
	
	if len(cntsColor) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle
        	c = max(cntsColor, key=cv2.contourArea)
        	((x, y), radius) = cv2.minEnclosingCircle(c)
	return radius
def getHomeColor(hsv):

	maskRed = cv2.inRange(hsv, lower_red, upper_red)
	maskBlue = cv2.inRange(hsv, lower_blue, upper_blue)
	maskGreen = cv2.inRange(hsv, lower_green, upper_green)
	maskOrange = cv2.inRange(hsv, lower_orange, upper_orange)
	maskYellow = cv2.inRange(hsv, lower_yellow, upper_yellow)-maskOrange-maskGreen
	maskWhite = cv2.inRange(hsv, lower_white, upper_white)-maskOrange
	
	domRad = 0;
	homeColor = 0

	redRadius = contourRadius(maskRed)
	if (domRad <= redRadius):
		domRad = redRadius
		homeColor = 1
	blueRadius = contourRadius(maskBlue)
	if (domRad <= blueRadius):
		domRad = blueRadius
		homeColor = 2
	greenRadius = contourRadius(maskGreen)
	if (domRad <= greenRadius):
		domRad = greenRadius
		homeColor = 3
	yellowRadius = contourRadius(maskYellow)
	if (domRad <= yellowRadius):
		domRad = yellowRadius
		homeColor = 4
	return homeColor

def makeMask(hsv, x):
	#1,2,3,4 = red,blue,green, or yellow
	if(x == 1):
		mask = cv2.inRange(hsv, lower_red, upper_red)
	if(x == 2):
		mask = cv2.inRange(hsv, lower_blue, upper_blue)
	if(x == 3):
		mask = cv2.inRange(hsv, lower_green, upper_green)
	if(x == 4):
		mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
	return mask

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--imagePath", required=True,
	help="Path to image to find dominant color of")

ap.add_argument("-k", "--clusters", default=4, type=int,
	help="Number of clusters to use in kmeans when finding dominant color")

args = vars(ap.parse_args())
#pts = deque(maxlen=args["buffer"])

image = cv2.imread(args['imagePath'])


#image = cv2.imread("pika.jpeg")
image = imutils.resize(image, width=600) #for faster processing?
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#bgr format
lower_red = np.array([0, 100, 100], dtype = "uint8")
upper_red = np.array([10, 255, 255], dtype = "uint8")

lower_blue = np.array([100,140,140], dtype = "uint8")
upper_blue = np.array([140,255,255], dtype = "uint8")


lower_green = np.array([29,86,6], dtype = "uint8")
upper_green = np.array([64,255,255], dtype = "uint8")

upper_white = np.array([131,255,255], dtype=np.uint8)
lower_white = np.array([0,0,190], dtype=np.uint8)

lower_yellow = np.array([20,100,175], dtype = "uint8")
upper_yellow = np.array([91,255,255], dtype = "uint8")

lower_orange = np.array([10,100,20], dtype=np.uint8)
upper_orange = np.array([25,255,255], dtype=np.uint8)
maskRed = cv2.inRange(hsv, lower_red, upper_red)
maskBlue = cv2.inRange(hsv, lower_blue, upper_blue)
maskGreen = cv2.inRange(hsv, lower_green, upper_green)
maskOrange = cv2.inRange(hsv, lower_orange, upper_orange)
maskYellow = cv2.inRange(hsv, lower_yellow, upper_yellow)-maskOrange-maskGreen
maskWhite = cv2.inRange(hsv, lower_white, upper_white)-maskOrange
	

num = getHomeColor(hsv)
homeColorMask = makeMask(hsv,num)

homeColorMask = cv2.erode(homeColorMask, None, iterations=2)
homeColorMask = cv2.dilate(homeColorMask, None, iterations=2)
cntsColor = cv2.findContours(homeColorMask.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)

cntsColor = imutils.grab_contours(cntsColor)#according to opencv version

center = None

    # only proceed if at least one contour was found
if len(cntsColor) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
	c = max(cntsColor, key=cv2.contourArea)
	((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)  #a moment is a particular weighted average of image pixel intensities
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # only proceed if the radius meets a minimum size
        if radius > 10: #need an approximate number for max radius
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(image, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(image, center, 5, (0, 0, 255), -1)


print(num)
#now test red, blue, yellow, green for contours
#print(domRad)
#print(homeColor)


#print(dom_color) 
cv2.drawContours(image, cntsColor, -1, (0,255,0), 3)
cv2.imshow("Original", image)
cv2.imshow("Home mask", homeColorMask)
#cv2.imshow("Yellow", maskYellow)
#cv2.imshow("Red", maskRed)
#cv2.imshow("Green", maskGreen)
#cv2.imshow("Blue", maskBlue)
#cv2.imshow("box of green", sub_image)



cv2.waitKey()
cv2.destroyAllWindows()
