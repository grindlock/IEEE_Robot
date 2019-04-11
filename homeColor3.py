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


def calculateDistance(x1,y1,x2,y2):  
	#returns distance in pixels of contour from the center of the screen
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
     return dist

def contourRadius(mask):
	#used to compare the 4 masks for the home color
	#gets the radius of the largest contour of the mask
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
	#returns a number from 1-4 indicating red, green, blue, or yellow main color
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
		homeColor = 3
	greenRadius = contourRadius(maskGreen)
	if (domRad <= greenRadius):
		domRad = greenRadius
		homeColor = 2
	yellowRadius = contourRadius(maskYellow)
	if (domRad <= yellowRadius):
		domRad = yellowRadius
		homeColor = 4
	return homeColor

def makeMask(hsv, x):
	#1,2,3,4 = red,green, blue, or yellow
	if(x == 1):
		mask = cv2.inRange(hsv, lower_red, upper_red)
	if(x == 3):
		mask = cv2.inRange(hsv, lower_blue, upper_blue)
	if(x == 2):
		mask = cv2.inRange(hsv, lower_green, upper_green)
	if(x == 4):
		mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
	return mask
def objectAbove(centerY,objectY):
	#if object is above center return 1 else 0
	if objectY < centerY:
		return 1
	return 0
def objectLeft(centerX, objectX):
	#if object is left of center return 1 else 0
	if objectX < centerX:
		return 1
	return 0    
#create a function to get contours. from mask
#the only problem is we need the radius and center of it as well.
def getContours(mask):
	
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
	contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
	contours = imutils.grab_contours(contours)#according to opencv version

	return contours

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--imagePath", required=True,
	help="Path to image to find dominant color of")

ap.add_argument("-k", "--clusters", default=4, type=int,
	help="Number of clusters to use in kmeans when finding dominant color")

args = vars(ap.parse_args())
#pts = deque(maxlen=args["buffer"])

image = cv2.imread(args['imagePath'])

image = imutils.resize(image, width=600) #for faster processing
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#bgr format
#lower_red = np.array([0, 100, 100], dtype = "uint8")
#upper_red = np.array([20, 255, 255], dtype = "uint8")

lower_red = np.array([160,20,20], dtype = "uint8")
upper_red = np.array([190, 255, 255], dtype = "uint8")

lower_blue = np.array([100,140,140], dtype = "uint8")
upper_blue = np.array([140,255,255], dtype = "uint8")


lower_green = np.array([29,86,6], dtype = "uint8")
upper_green = np.array([64,255,255], dtype = "uint8")

upper_white = np.array([131,255,255], dtype=np.uint8)
lower_white = np.array([0,0,190], dtype=np.uint8)

#lower_yellow = np.array([20,100,175], dtype = "uint8")
#upper_yellow = np.array([91,255,255], dtype = "uint8")

lower_yellow = np.array([0, 155, 121])
upper_yellow = np.array([253, 252, 253])

lower_orange = np.array([10,100,20], dtype=np.uint8)
upper_orange = np.array([25,255,255], dtype=np.uint8)

maskRed = cv2.inRange(hsv, lower_red, upper_red)
maskBlue = cv2.inRange(hsv, lower_blue, upper_blue)
maskGreen = cv2.inRange(hsv, lower_green, upper_green)
maskOrange = cv2.inRange(hsv, lower_orange, upper_orange)
maskYellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
maskWhite = cv2.inRange(hsv, lower_white, upper_white)-maskOrange
	

num = getHomeColor(hsv) #get number of home color
homeColorMask = makeMask(hsv,num)#make mask of home color

cntsColor = getContours(homeColorMask)

center = None

    # only proceed if at least one contour was found
if len(cntsColor) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
	c = max(cntsColor, key=cv2.contourArea)
	((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)  #a moment is a particular weighted average of image pixel intensities
	#if M["m00"] == 0:#prevents crash if small contours are present
	#	continue
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # only proceed if the radius meets a minimum size
        if radius > 10: #need an approximate number for max radius
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(image, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(image, center, 5, (0, 0, 255), -1)

(h, w) = hsv.shape[:2] #w:image-width and h:image-height
    #the center of an image is
imageCenterX = w//2
imageCenterY = y//2
objX = center[0]
objY = center[1]
#distance in pixels from center of image to center of contour
distanceFromCenterImage = (imageCenterX, imageCenterY, center[0], center[1])

left = -1
above = -1
#left and above is 1 if true 0 if false, -1 if error
left = objectLeft(imageCenterX, objX)
above = objectAbove(imageCenterY, objY)

#attempt to find center of line by finding the second largest white contour

cntsWhite = getContours(maskWhite)


#sort contours by area:
areaArray = []
for i, c in enumerate(cntsWhite):
    area = cv2.contourArea(c)
    areaArray.append(area)

sortedCnts = sorted(zip(areaArray, cntsWhite), key=lambda x: x[0], reverse=True)

centerWhite = None

if len(cntsWhite) > 2: #if camera sees wall and line


	#find the nth largest contour [n-1][1], in this case 2
	c = sortedCnts[1][1]
	((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(cntsWhite)  #a moment is a particular weighted average of image pixel intensities
	'''if M["m00"] == 0:#prevents crash if small contours are present
            continue
	'''        
	centerWhite = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
#now we have the x, y of the center of the white line.
if len(cntsWhite) > 2:
	lineX = centerWhite[0]
	lineY = centerWhite[1]
else:#in the case it isn't detected
	lineX = -1
	lineY = -1


#printing for testing purposes
print(left, above) #position of obj relative to center
print(num) #home color
print(lineX, lineY) # line position
#cv2.drawContours(image, cntsColor, -1, (0,255,0), 3)#draw home color contour
cv2.drawContours(image, sortedCnts[1][1], -1, (0,255,0), 3)#draw white color contour
cv2.imshow("Original", image)
cv2.imshow("Home mask", maskWhite)
#cv2.imshow("Yellow", maskYellow)
#cv2.imshow("Red", maskRed)
#cv2.imshow("Green", maskGreen)
#cv2.imshow("Blue", maskBlue)


cv2.waitKey()
cv2.destroyAllWindows()
