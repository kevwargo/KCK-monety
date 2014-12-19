#!/usr/bin/python2

import sys, os
import cv2
# from SimpleCV import *
import numpy as np

Templates = {}

def loadTemplates(path):
    for f in os.listdir(os.path.join(path, 'templates')):
        Templates[os.path.splitext(f)[0]] = parseFile(os.path.join(path, os.path.join('templates', f)))

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

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
def findCircles(contours, minSize, maxSize):
    circles = [cv2.minEnclosingCircle(cv2.convexHull(cnt[:, 0, :], returnPoints=True)) for cnt in contours]
    for c in circles[:]:
        if c[1] < minSize or c[1] > maxSize:
            circles.remove(c)
        for o in circles[:]:
            if o[1] > c[1] and distance(c[0], o[0]) < (c[1] + o[1]):
                try: circles.remove(c)
                except: pass
    return circles

def matchCoin(img):
    for tname, timg in Templates.items():
        orb = cv2.ORB()
        kpimg, desimg = orb.detectAndCompute(img,None)
        kptmpl, destmpl = orb.detectAndCompute(timg,None)
        bf = cv2.BFMatcher()
        # print(desimg, destmpl, timg)
        matches = bf.knnMatch(desimg, destmpl, k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        print(tname, len(good))
    # return len(good)

def processImage(src):
    color = src
    gray = cv2.cvtColor(color, cv2.cv.CV_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 100)
    # dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    img = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13, 13)))
    # edges = np.copy(img)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
    # empty = np.zeros(gray.shape, dtype=np.uint8)
    # cv2.circle(empty, (len(gray[0]) / 2, len(gray) / 2), len(gray[0]) / 15, 255)
    # circleSample = cv2.findContours(empty, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
    drawing = np.zeros(gray.shape + (3,))
    for (x, y), r in findCircles(contours, len(img) / 30, len(img) / 5):
        x = int(x)
        y = int(y)
        r = int(r)
        matchCoin(color[y-r:y+r, x-r:x+r])
        # drawing = color[y-r:y+r, x-r:x+r]
        # cv2.circle(color, tuple(map(int, coin[0])), int(coin[1]), (0, 255, 255), 3)
    # cv2.HoughCircles(drawing, )
    # print([cv2.arcLength(cnt, False) for cnt in contours])
    return drawing

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
        loadTemplates(os.path.realpath(os.path.dirname(sys.argv[0])))
        main(sys.argv[1:])
