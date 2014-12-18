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
                               param1=60, param2=40,
                               minRadius=150,
                               maxRadius=1500)
    for i in circles[0,:]:
        cv2.circle(tgt, (i[0], i[1]), i[2], Color.RED, 3)

def GaussTest(src, tgt):
    img = cv2.GaussianBlur(src, (5, 5), 2, sigmaY=2)
    cv2.imwrite(tgt, img)
    
        
def CannyTest(src, res):
    cimg = cv2.imread(src, 1)
    cfiltered = cv2.pyrMeanShiftFiltering(cimg, 20, 20, 1)
    filtered = cv2.cvtColor(cimg, cv2.cv.CV_BGR2GRAY)
    # filtered = filtered.astype(float)
    # filtered *= 1./255.
    # print(filtered)
    blurred = cv2.GaussianBlur(filtered, (5, 5), 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_8U)
    # sharpened = (laplacian.astype(float)/255.) * 1.5
    # edges = cv2.Canny(filtered, 100, 50, L2gradient=True)
    # target = Image(edges.transpose())
    findCircles(laplacian, cimg)
    # cv2.circle(edges, (600, 200), 100, Color.RED, 3)
    cv2.imwrite(res, cimg)
    # cv2.imwrite(res, filtered)

def ParseImage(src, res):
    pass


if __name__ == '__main__':
    # houghCirclesTest(sys.argv[1], sys.argv[2])
    CannyTest(*sys.argv[1:3])
    # ParseImage(sys.argv[1], sys.argv[2])
