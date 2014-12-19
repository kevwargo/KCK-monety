#!/usr/bin/python2

import sys
import cv2
# from SimpleCV import *
import numpy as np

def hsv2rgb(h, s, v):
    # 100-110-010-011-010-101-100
    rgb = generateGradient(h, (1,0,0), (1,1,0), (0,1,0),
                                (0,1,1), (0,0,1), (1,0,1), (1,0,0))
    for i, ch in enumerate(rgb):
        if ch < 1.0:
            rgb[i] = max(rgb) - s*(max(rgb) - ch)
    for i, ch in enumerate(rgb):
        if ch > 0.0:
            rgb[i] *= v
    rgb = [int(255 * c) for c in rgb]
    return rgb
    # return list(map(lambda c: c*256, rgb))
    # return (h, s, v)

def generateGradient(v, *colors):
    count = len(colors) - 1
    pos = int(np.floor(v * count))
    left = colors[pos]
    try:
        right = colors[pos + 1]
    except IndexError:
        right = left
    offset = v*count - pos
    return [l + (r-l)*offset for l, r in zip(left, right)]

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

def drawPoint(img, x, y, color):
    x = int(round(x))
    y = int(round(y))
    cv2.line(img, (x, y), (x, y), color)

# def findCircles(contours, minSize,):
    

def processImage(src):
    color = src
    gray = cv2.cvtColor(color, cv2.cv.CV_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 100)
    # dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    img = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13, 13)))
    edges = np.copy(img)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
    # empty = np.zeros(gray.shape, dtype=np.uint8)
    # cv2.circle(empty, (len(gray[0]) / 2, len(gray) / 2), len(gray[0]) / 15, 255)
    # circleSample = cv2.findContours(empty, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
    # drawing = np.zeros(gray.shape + (3,))
    for cnt in contours:
        # if cv2.arcLength(cnt, False) > 200:
        #     pointcount = float(len(cnt))
        #     # cv2.drawContours(drawing, [cnt], 0,
        #     #                   # (np.random.randint(0, 256),
        #     #                   #  np.random.randint(0, 256),
        #     #                   #  np.random.randint(0, 256)),
        #     #                   255,
        #     #                   1)
        #     for i, vert in enumerate(cnt):
        #         drawPoint(drawing, vert[0][0], vert[0][1],
        #                   hsv2rgb(*generateGradient(i / pointcount, (0.2,1,1), (1,1,1))))
        hull = cv2.convexHull(cnt[:, 0, :], returnPoints=True)
        coin = cv2.minEnclosingCircle(hull)
        # cv2.circle(color, tuple(map(int, coin[0])), int(coin[1]), (0, 255, 0), 3)
        cv2.circle(edges, tuple(map(int, coin[0])), int(coin[1]), 255, 3)
        # cv2.drawContours(drawing, [hull], 0, (255, 255, 255))
    # cv2.HoughCircles(drawing, )
    # print([cv2.arcLength(cnt, False) for cnt in contours])
    return edges

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
