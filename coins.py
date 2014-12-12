#!/usr/bin/python2

import sys
from itertools import combinations
from SimpleCV import *
# import cv2
# import cv2.cv as cv
import numpy as np

coins = {
    '1gr':  15.5,
    '2gr':  17.5,
    '5gr':  19.5,
    '10gr': 16.5,
    '20gr': 18.5,
    '50gr': 20.5,
    '1zl':  23.0,
    '2zl': 21.5,
    '5zl': 24.0,
    }

diameters = [(d, n) for n, d in coins.items() if n in ('10gr', '50gr', '1zl')]
pairs = [(i, j) if i < j else (j, i) for i, j in combinations(diameters, 2)]

def display(pairs):
    for (d1, n1), (d2, n2) in pairs:
        print(d1 / d2, n1, n2)

def findCircles(src, tgt):
    circles = cv2.HoughCircles(src, cv.CV_HOUGH_GRADIENT, 1, 200,
                               param1=100, param2=50,
                               minRadius=150,
                               maxRadius=1500)
    for i in circles[0,:]:
        cv2.circle(tgt, (i[0], i[1]), i[2], Color.GREEN, 3)

def GaussTest(src, tgt):
    img = cv2.GaussianBlur(src, (5, 5), 2, sigmaY=2)
    cv2.imwrite(tgt, img)
    
        
def CannyTest(src, res):
    img = cv2.imread(src, 0)
    edges = cv2.Canny(img, 80, 40, L2gradient=True)
    target = Image(edges.transpose())
    findCircles(edges, target)
    target.save(res)
    # cv2.imwrite(res, target)

def ParseImage(src, res):
    cimg = cv2.imread(src, cv2.CV_LOAD_IMAGE_COLOR)
    cfiltered = cv2.pyrMeanShiftFiltering(cimg, 10, 10, 1)
    filtered = cv2.cvtColor(cfiltered, cv2.cv.CV_BGR2GRAY)
    print(filtered)
    # findCircles(filtered, cimg)
    # cv2.circle(cimg, (200, 500), 50, Color.GREEN, 3)
    
    # cv2.imwrite(res, cimg)
    
    
if __name__ == '__main__':
    # houghCirclesTest(sys.argv[1], sys.argv[2])
    ParseImage(sys.argv[1], sys.argv[2])
