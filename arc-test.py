#!/usr/bin/python2

import sys
import numpy as np
import cv2

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

def main(outfile):
    img = np.zeros((800, 1200), dtype=np.uint8)
    xcenter = 600
    ycenter = 400
    radius = 380
    for angle in np.radians(np.linspace(0, 360, 8)):
        # cv2.circle(img, (xcenter + int(radius * np.cos(angle)),
        #                  ycenter + int(radius * np.sin(angle))),
        #                 2, 255, 2)
        cv2.line(img, (xcenter + int((radius - 5) * np.cos(angle)),
                       ycenter + int((radius - 5) * np.sin(angle))),
                      (xcenter + int((radius + 5) * np.cos(angle)),
                       ycenter + int((radius + 5) * np.sin(angle))),
                      255)
    cv2.circle(img, (xcenter, ycenter), radius, 255)
    cimg = np.zeros(img.shape + (3,), dtype=np.uint8)
    cimg[:, :, :] = 255
    # cv2.circle(cimg, (xcenter, ycenter), radius, (255, 0, 0))
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print(len(contours))
    pointcount = float(len(contours[0]))
    # for i, cnt in enumerate(contours[0]):
    #     cv2.circle(cimg, (cnt[0][0], cnt[0][1]), 0,
    #                hsv2rgb(*generateGradient(i / pointcount, (0.5, 1, 1), (1, 1, 1))), 1)
    hit = 0
    # radius -= 0.
    # for pt in contours[0]:
    #     x = pt[0][0] - xcenter
    #     y = pt[0][1] - ycenter
    #     # print radius**2)
    #     # cv2.line(cimg, )
    #     if abs(((float(x)**2) + (float(y)**2)) - (float(radius)**2)) < 100:
    #         cv2.line(cimg, (pt[0][0],pt[0][1]), (pt[0][0],pt[0][1]), (0, 255, 0))
    #         hit += 1
    #     else:
    #         # print(x, y, x**2, y**2, radius**2,
    #         #       np.sqrt(abs(((float(x)**2) + (float(y)**2)) - (float(radius)**2))))
    #         cv2.line(cimg, (pt[0][0],pt[0][1]), (pt[0][0],pt[0][1]), (0, 0, 255))
    # cv2.line(cimg, (-46 + xcenter, -184 + ycenter), (-46 + xcenter, -184 + ycenter), (255, 0, 0))
    # drawCircleByAngles(cimg, xcenter, ycenter, radius)
    drawCircle(cimg, xcenter, ycenter, radius)
    print(hit, pointcount, hit / pointcount)
    cv2.imwrite(outfile, cimg)

def drawCircleByAngles(img, xc, yc, r):
    for a in np.arange(0, 361, 0.2):
        cv2.line(img, (int(round(xc + np.cos(a)*r)), int(round(yc + np.sin(a)*r))),
                      (int(round(xc + np.cos(a)*r)), int(round(yc + np.sin(a)*r))),
                      (0, 0, 0))


def drawCircle(img, xc, yc, r):
    X = np.linspace(-r, r, 1024)
    Y1 = np.sqrt(r**2 - X**2)
    Y2 = -np.sqrt(r**2 - X**2)
    for x, y in zip(X, Y1):
        cv2.line(img, (int(round(xc + x)), int(round(yc + y))),
                      (int(round(xc + x)), int(round(yc + y))), (0,0,0))
    for x, y in zip(X, Y2):
        cv2.line(img, (int(round(xc + x)), int(round(yc + y))),
                      (int(round(xc + x)), int(round(yc + y))), (0,0,0))
    # cv2.imwrite('result.png', img)

if __name__ == '__main__':
    main(sys.argv[1])
    # drawArc()
