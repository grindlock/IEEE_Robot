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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
def findLines(image):
	#this method might only find straight lines not curved ones, but not able to test anyway
	#use white mask and image to create new image with only white
        masked = cv2.bitwise_and(image, image, mask = maskWhite)

	# Convert to grayscale here.
	grayImage = cv2.cvtColor(masked, cv2.COLOR_RGB2GRAY)
	#apply gaussian blur to smooth edges
	blurImage = cv2.GaussianBlur(grayImage, (11, 11), 0)
	# Call Canny Edge Detection here. 
'''	
These two threshold values are empirically determined. Basically, you will need to define them by trials and errors.
I first set the low_threshold to zero and then adjust the high_threshold. If high_threshold is too high, you find no edges. If high_threshold is too low, you find too many edges. Once you find a good high_threshold, adjust the low_threshold to discard the weak edges (noises) connected to the strong edges.
'''
	low_threshold = 100
	high_threshold = 100
	cannyed_image = cv2.Canny(blurImage, low_threshold, high_threshold)
	#now apply Hough transforms to detect the lines from image
	lines = cv2.HoughLinesP(#need to adjust these parameters in testing
	    cropped_image,
	    rho=6,
	    theta=np.pi / 60,
	    threshold=160,
	    lines=np.array([]),
	    minLineLength=40,
	    maxLineGap=25
	)
	return lines
	#Each line is represented by four numbers, which are the two endpoints of the detected line segment, like so
	#[x1, y1, x2, y2]
def average_slope_intercept(lines):
    lines    = [] # (slope, intercept)
    weights  = [] # (length,)
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2==x1:
                continue # ignore a vertical line
            slope = (y2-y1)/(x2-x1)
            intercept = y1 - slope*x1
            length = np.sqrt((y2-y1)**2+(x2-x1)**2)
 
    # add more weight to longer lines    
    lane  = np.dot(weights, lines) /np.sum(weights)  if len(weights) >0 else None
    
    return lane # (slope, intercept)

def make_line_points(y1, y2, line):
    """
    Convert a line represented in slope and intercept into pixel points
    """
    if line is None:
        return None
    
    slope, intercept = line
    
    # make sure everything is integer as cv2.line requires it
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    
    return ((x1, y1), (x2, y2))
def lane_lines(image, lines):
    lane = average_slope_intercept(lines)
    
    y1 = image.shape[0] # bottom of the image
    y2 = 0         # top of image

    line  = make_line_points(y1, y2, lane)
    
    return line
def fLines(image):
	#function that combines the lines functions above
	lines = findlines(image)
	line = lane_lines(image_lines) # returns ((x1, y1), (x2, y2)) of the biggest line(I think)
	return line

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--imagePath", required=True,
	help="Path to image to find dominant color of")

ap.add_argument("-k", "--clusters", default=4, type=int,
	help="Number of clusters to use in kmeans when finding dominant color")

args = vars(ap.parse_args())
#pts = deque(maxlen=args["buffer"])

image = cv2.imread(args['imagePath'])

image = imutils.resize(image, width=600) #for faster processing
image = cv2.GaussianBlur(image, (11, 11), 0)
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

line = fLines(hsv) #should find the biggest line, maybe of form ((x1, y1), (x2, y2))

'''
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
	'''#if M["m00"] == 0:#prevents crash if small contours are present
            #continue
	'''        
	centerWhite = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
#now we have the x, y of the center of the white line.
if len(cntsWhite) > 2:
	lineX = centerWhite[0]
	lineY = centerWhite[1]
else:#in the case it isn't detected
	lineX = -1
	lineY = -1
'''

#printing for testing purposes
print(left, above) #position of obj relative to center
print(num) #home color
#print(lineX, lineY) # line position
#cv2.drawContours(image, cntsColor, -1, (0,255,0), 3)#draw home color contour
cv2.drawContours(image, sortedCnts[1][1], -1, (0,255,0), 3)#draw white color contour
cv2.imshow("Original", image)
cv2.imshow("Home mask", homeColorMask)
#cv2.imshow("Yellow", maskYellow)
#cv2.imshow("Red", maskRed)
#cv2.imshow("Green", maskGreen)
#cv2.imshow("Blue", maskBlue)


cv2.waitKey()
cv2.destroyAllWindows()
