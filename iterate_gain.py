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
img0 = np.arange(6).reshape(3,2)
img1 = img0 * np.array([2,3])
print (img0)
print (img1)

def f(x,img0,img1):
    x = np.reshape(x,-1)
    pre_dis = x * img0
    loss = np.sum((pre_dis - img1)**2) / 2
    return loss

# a = optimize.fmin_cg(f,[2.0,3.0], args=(img0,img1),constraints=cons)
a = optimize.fmin(f,[1.0,1.0], args=(img0,img1))
print (a)
