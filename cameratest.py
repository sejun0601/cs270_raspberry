from picamera import PiCamera
from time import sleep

camera = PiCamera()
sleep(1)
camera.capture('capture.jpg')