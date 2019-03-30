#
# This is a space to experiment to learn
# about the RPI GPIO and ultrasonic sensors
#
# Taylor Barnes
# IEEE @ UCF
#

#https://sourceforge.net/p/raspberry-gpio-python/wiki/Examples/
#https://bitbucket.org/theconstructcore/morpheus_chair
#https://www.modmypi.com/blog/hc-sr04-ultrasonic-range-sensor-on-the-raspberry-pi
import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)



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
	print("Out of Bounds")

#otherwise return how far away the object is
else:
	print(distance)