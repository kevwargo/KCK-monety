#!/usr/bin/python2

import sys
import cv2
# from SimpleCV import *
import numpy as np

def asFloat(img):
    return np.array(img.astype(float) / 255., dtype=np.float64)

def asUInt8(img):
    return np.array(img * 255., dtype=np.uint8)

def gammaCorrect(img, gamma):
    return asUInt8(asFloat(img) ** (1. / gamma))

def roundto(a, to):
    if isinstance(a, int):
        a = float(a)
    elif isinstance(a, np.ndarray):
        a = a.astype(float)
    return to * np.around(a / to)

def processImage(src):
    color = src
    gray = cv2.cvtColor(color, cv2.cv.CV_BGR2GRAY)
    # img = cv2.fastNlMeansDenoisingColored(color,
    #                                 h=3,
    #                                 hColor=5
    #                                 )
    edges = cv2.Canny(gray, 50, 200)
    # dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    img = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    empty = np.zeros(gray.shape)
    for cnt in contours:
        if cv2.arcLength(cnt, False) > 200:
            cv2.drawContours(empty, [cnt], 0,
                              # (np.random.randint(0, 256),
                              #  np.random.randint(0, 256),
                              #  np.random.randint(0, 256)),
                              255,
                              5)
    cv2.HoughCircles(empty, )
    # print([cv2.arcLength(cnt, False) for cnt in contours])
    return empty

def parseFile(filename):
    # img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    return img

def saveImage(img, filename):
    cv2.imwrite(filename, img)

def resize(img, tw, th):
    sh, sw = img.shape[:2]
    aspratio = float(sw) / float(sh)
    if aspratio > float(tw) / float(th):
        w = tw
        h = int(round(tw / aspratio))
    else:
        h = th
        w = int(round(th * aspratio))
    return cv2.resize(img, (w, h))

def main(args):
    # print(processImage(parseFile(args[0])), args[1])
    img = processImage(parseFile(args[0]))
    saveImage(img, args[1])
    # cv2.imshow('res', resize(img, 1365, 740))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1:])
