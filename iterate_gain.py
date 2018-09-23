import os
import scipy
from scipy import optimize
from skimage import io, color
import numpy as np
import cv2
from skimage.transform import resize
from utils import *



est = np.load('save_dir_new/000_00.npy')
concat = cv2.imread('save_dir_new/000_00.png')
h,w,c = concat.shape
w_cut = w //4
img0 = concat[:,0:w_cut,:]
img2 = concat[:,2*w_cut:3*w_cut,:]
raise

def f(x,img0,img1):
    x = np.reshape(x,-1)
    img0 = np.clip(img0 * x[::-1], 0, 255)
    # img0 = color.rgb2lab(img0)
    # img1 = color.rgb2lab(img1)
    loss = np.sum((img0 - img1)**2) / 2
    return loss

gain = np.array([2.531,1.0,1.793]) # 000_00.tiff
# gain = np.array([2.848,1.0,1.656]) #000_03.tiff
# img = cv2.imread('input.png')
# concat = cv2.imread('000_03.tiff').astype(np.float32)
concat = cv2.imread('000_00.tiff').astype(np.float32)
h,w,c = concat.shape
w_cut = w //3
img = concat[:,0:w_cut,:]

img_gt_pure = np.clip(img * gain[::-1], 0, 255)
a = optimize.fmin(f, [1.0, 1.0, 1.0], args= (img, img_gt_pure))
print (a)
img_gt = concat[:, w_cut: 2*w_cut, :]
b = optimize.fmin(f, [1.0, 1.0, 1.0], args = (img, img_gt))
print (b)
img_est = concat[:,2*w_cut:3*w_cut,:]
c = optimize.fmin(f, [1.0, 1.0, 1.0], args = (img, img_est))
print (c)

print ('angluar of pure gt :', angular_error(np.array(a), gain))
print ('angluar of gt :', angular_error(np.array(b), gain))
print ('angluar of est :', angular_error(np.array(c), gain))

# est_gain = np.array([ 2.72230778,0.99272707,1.69406263])
# img_a = img_gt_pure
# img_b = np.clip(img * est_gain[::-1], 0, 255)
# cv2.imwrite('a.png', img_a)
# cv2.imwrite('b.png', img_b)
