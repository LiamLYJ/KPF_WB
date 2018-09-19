import os
import json
import sys
import utils
import numpy as np
import cv2
from skimage.io import imread
from skimage import transform
from scipy.misc import imsave


if __name__ == '__main__':

    # fn = '111MSDCF_DSC07730_108MSDCF_DSC07506'
    # fn = '108MSDCF_DSC07087_111MSDCF_DSC07977'
    # fn = '111MSDCF_DSC07750_109MSDCF_DSC06798'
    # fn = '112MSDCF_DSC08800_111MSDCF_DSC07916'
    # fn = '105MSDCF_DSC03577_112MSDCF_DSC09082'
    fn = '106MSDCF_DSC04946_106MSDCF_DSC04946'

    dir = './tmp_save_ex'
    file_np = os.path.join(dir, fn + '.npy')
    tmp = np.load(file_np)

    size = tmp.shape[1]

    img0 = tmp[:size,:,:]
    img2 = tmp[size:,:,:]
    img2 = np.clip(img2, 0,500)
    print ('min ref:', np.min(img2))
    print ('max ref:', np.max(img2))

    data_dir = '/home/lyj/Downloads/Sony/preprocessed'
    json_file = os.path.join(dir, fn + '.json')

    concat_original, scale_h, scale_w = utils.get_original(data_dir= data_dir, json_file = json_file)

    gain_box_pure = utils.gain_fitting(img0, img2, is_pure = True)
    pure = utils.apply_gain_box(concat_original, gain_box_pure, scale_h, scale_w)
    imsave(fn+'_big_pure.png', pure)

    gain_box, clus_img, clus_labels = utils.gain_fitting(img0, img2, is_local = True, n_clusters =2, gamma = 4.0, with_clus = True)
    imsave(fn+'_clus.png', clus_img)
    big = utils.apply_gain_box(concat_original, gain_box, scale_h, scale_w)
    imsave(fn + '_big.png', big)
