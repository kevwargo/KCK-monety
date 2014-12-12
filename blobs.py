#!/usr/bin/env python2

import sys
import SimpleCV as cv

def main(name):
    img = cv.Image(name)
    img = img.hueDistance(cv.Color.YELLOW)
    img.show()
    raw_input()
    # blobs = img.findBlobs(minsize=100)
    # coins = [b for b in blobs if b.isCircle(0.1)]
    # print('len', len(coins), coins)

if __name__ == '__main__':
    main(sys.argv[1])
