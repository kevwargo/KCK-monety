#!/usr/bin/env python2

from SimpleCV import Camera

cam = Camera()

while 1:
    cam.getImage().show()
    
