#!/usr/bin/env python3

import sys
from skimage import data, morphology, measure
from skimage import filter as skifilter
from scipy import ndimage
# import matplotlib.pyplot as plt
import numpy as np

def parseFile(fname):
    img = data.imread(fname, as_grey=True)
    img **= 0.4
    img = skifilter.canny(img, sigma=3)
    img = morphology.dilation(img, morphology.disk(4))
    return img

# def showImage(img, fig=None, out=None):
#     if not fig:
#         fig = plt.figure(figsize=(5,5))
#     if not fig.axes:
#         ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
#     else:
#         ax = fig.axes[0]
#     ax.axis('off')
#     ax.imshow(img)
#     if out:
#         fig.savefig(out)
#     else:
#         fig.show()


if __name__ == '__main__':
    img = parseFile(sys.argv[1])
    # if len(sys.argv) < 3:
    #     out = None
    # else:
    #     out = sys.argv[2]
    # showImage(img, out=out)

