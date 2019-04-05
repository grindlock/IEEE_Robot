#!/usr/bin/env python

#
# This node should relay information to
# ROS if the ultrasonic sensor detects
# something in the robot's path
#
# Taylor Barnes
# IEEE @ UCF
#

#license removed for brevity
import rospy
from std_msgs.msg import String
import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

def talker():
    pub = rospy.Publisher('chatter', String, queue_size = 10) #output a string 10 times
    rospy.init_node('ultrasonic', anonymous = True) #name the node
    rate = rospy.Rate(2) #twice a second
	
    while not rospy.is_shutdown():
		#if there is nothing in our way, we output CLEAR
		#if there our path is blocked, we output BLOCK
        clearStr = "CLEAR"
		blockStr = "BLOCK"


		#global variables
		maxDistance = 50
		kP = 17150

		#pinout
		triggerPin = 15
		echoPin = 16
		GPIO.setup(triggerPin, GPIO.OUT)
		GPIO.setup(echoPin, GPIO.IN)

		#shoot a quick pulse
		GPIO.output(triggerPin, 1)
		time.sleep(0.00001)
		GPIO.output(triggerPin, 0)

	
		#wait for the sensor to get something back
		#remember the timestamps when it starts looking
		#and when it gets something back
		while not GPIO.input(echoPin):
			pulseStart = time.time()
		while GPIO.input(echoPin):
			pulseEnd = time.time()
	
		#now calculate the time elapsed
		pulseDuration = pulseEnd - pulseStart
		distance = pulseDuration * kP

		#if the distance is farther away than we care to measure
		#say that there's nothing in front of us
		if distance > maxDistance:
			pub.publish(clearStr)

		#otherwise return how far away the object is
		else:
			pub.publish(blockStr)
				
				
				
	rate.sleep()