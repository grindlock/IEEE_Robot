
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import homeColor3
import rospy
from std_msgs.msg import String
import math

def calculateDistance(x1,y1,x2,y2):  
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
     return dist
def makeMask(hsv, x):
	#1,2,3,4 = red,blue,green, or yellow
	if(x == 1):
		mask = (hsv, lower_red, upper_red)
	if(x == 2):
		mask = (hsv, lower_blue, upper_blue)
	if(x == 3):
		mask = (hsv, lower_green, upper_green)
	if(x == 4):
		mask = (hsv, lower_yellow, upper_yellow)
	return mask
def objectAbove(centerY,objectY):
	#if object is above center return 1 else 0
	if objectY > centerY:
		return 1
	return 0
def objectLeft(centerX, objectX):
	#if object is left of center return 1 else 0
	if objectX < centerX:
		return 1
	return 0    

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
                help="max buffer size")

#ap.add_argument("-k", "--clusters", default=4, type=int,
	#help="Number of clusters to use in kmeans when finding dominant color")

args = vars(ap.parse_args())
# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points


lower_red = np.array([0, 100, 100], dtype = "uint8")
upper_red = np.array([10, 255, 255], dtype = "uint8")

lower_blue = np.array([90,50,50], dtype = "uint8")
upper_blue = np.array([130,255,255], dtype = "uint8")

lower_green = np.array([35,0,0], dtype = "uint8")
upper_green = np.array([85,255,255], dtype = "uint8")

upper_white = np.array([131,255,255], dtype=np.uint8)
lower_white = np.array([0,0,190], dtype=np.uint8)

lower_yellow = np.array([20,100,175], dtype = "uint8")
upper_yellow = np.array([91,255,255], dtype = "uint8")

lower_orange = np.array([10,100,20], dtype=np.uint8)
upper_orange = np.array([25,255,255], dtype=np.uint8)

pts = deque(maxlen=args["buffer"])
# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
    vs = VideoStream(src=0).start()
    # otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])
    # allow the camera or video file to warm up

time.sleep(2.0)

# keep looping
while True:
    # grab the current frame
    frame = vs.read()
    # handle the frame from VideoCapture or VideoStream
    frame = frame[1] if args.get("video", False) else frame
    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if frame is None:
        break
    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = imutils.resize(frame, width=600) #for faster processing?
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	
    #run home color now to find it, it returns 3 numbers which is the rgb value
    #cv2.imWrite("9000.png",blurred)
    dom_color = homeColor2.getHomeColor(hsv) #gets a number 1 to 4 that is r,g,b, or y
    '''
    b = dom_color[0]
    g = dom_color[1]
    r = dom_color[2]

    #create a mask using that rgb value by upper bound tint and lower bound shade
#newR = currentR * (1 - shade_factor)
#newG = currentG + (255 - currentG) * tint_factor
    shadeFactor = .35
    tintFactor = .35
    tintRed = r + (255-r)*tintFactor
    tintBlue = b + (255-b)*tintFactor
    tintGreen = g + (255-g)*tintFactor

    shadeRed = r * (1-shadeFactor)
    shadeBlue = b * (1-shadeFactor)
    shadeGreen = g * (1-shadeFactor)


    dom_lower = np.array([shadeBlue, shadeGreen, shadeRed], dtype = "uint8")#darker
    dom_upper = np.array([tintBlue,tintGreen, tintRed], dtype = "uint8")#lighter
    homeColorMask = cv2.inRange(hsv, dom_lower, dom_upper)
    '''
    
    
    # construct a mask for the home color , then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    homeColorMask = homeColor2.makeMask(hsv, dom_color)
    #homeColorMask = np.all(blurred == dom_color,axis=-1)
    homeColorMask = cv2.erode(homeColorMask, None, iterations=2)
    homeColorMask = cv2.dilate(homeColorMask, None, iterations=2)
    
    
    
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cntsColor = cv2.findContours(homeColorMask.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    #cntsGreen = cv2.findContours(maskGreen.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
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
        if radius > 10 and radius < 200: #need an approximate number for max radius
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            # update the points queue
    pts.appendleft(center)
   
    #distance of object from center.. in pixels

    #centroid or center of image
    (h, w) = hsv.shape[:2] #w:image-width and h:image-height
    #the center of an image is
    imageCenterX = w//2
    imageCenterY = y//2
    
    #distance in pixels from center of image to center of contour
    distanceFromCenterImage = (imageCenterX, imageCenterY, center[0], center[1])
    
    #might need radius
    #object is to the right
    #if imageCenterX > center[0]:
    #elseif center[0]-imageCenterX = 0: in front of robot
    #else to the left



   #this draws the contrails of the ball

    # loop over the set of tracked points
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue
        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # show the frame to our screen
    #cv2.imshow("Frame", frame)
    #key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    '''if key == ord("q"):
        break
    '''
    # if we are not using a video file, stop the camera video stream
if not args.get("video", False):
    vs.stop()
    # otherwise, release the camera
else:
    vs.release()
    # close all windows
cv2.destroyAllWindows()
