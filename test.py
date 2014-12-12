#!/usr/bin/env python2

from SimpleCV import *
import time, sys

def main(fname, result):
    img = Image(fname)
    img = img.colorDistance((160, 160, 0))
    
    img.save(result)
    # immatrix = img.getGrayNumpy()
    # immatrix = immatrix.transpose()
    # # immatrix = cv2.pow(immatrix, 2)
    # print(immatrix[24][10])
    # # thr = img.getEmpty(1)
    # thr = cv2.adaptiveThreshold(immatrix, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
    # cv2.imwrite(result, thr)
    # img.save(result)
    # time.sleep(30)

def netex():
    
    # Make a function that does a half and half image.
    def halfsies(left,right):
        result = left
        # crop the right image to be just the right side.
        crop = right.crop(right.width/2.0,0,right.width/2.0,right.height)
        # now paste the crop on the left image.
        result = result.blit(crop,(left.width/2,0))
        # return the results.
        return result
    # Load an image from imgur.
    img = Image('http://i.imgur.com/lfAeZ4n.png')
    # create an edge image using the Canny edge detector
    # set the first threshold to 160
    output = img.edges(t1=160)
    # generate the side by side image.
    result = halfsies(img,output)
    # show the results.
    # result.show()
    # save the output images.
    result.save('juniperedges.png')

    
if __name__ == '__main__':
    # netex()
    main(sys.argv[1], sys.argv[2])
    # example()
