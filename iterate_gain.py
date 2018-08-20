import os
import scipy
from scipy import optimize
from skimage import io, color
import numpy as np

# file_0 = ''
# file_1 = ''
# img_0 = io.read(file_0)
# img_1 = io.read(file_1)
# img_0 = color.rgb2lab(img_0)
# img_1 = color.rgb2lab(img_1)
img0 = np.arange(6).reshape(2,3).astype(np.float32)
img1 = np.arange(6).reshape(2,3).astype(np.float32)
img1[0,:] *= 2
img1[1,:] *= 3

def f(x,img0,img1):
    img0[0,:] *= x[0]
    img1[1,:] *= x[1]
    return np.sum(np.abs(img0-img1))
print (img0, img1)
a = optimize.fmin_cg(f,[1.0,1.0], args=(img0,img1))
print (a)


#
# def f(x, img0,img1):
#     return abs(img0*x - img1)
#
# a = optimize.fmin_cg(f,[0], args=(10,2))
# print(a)
