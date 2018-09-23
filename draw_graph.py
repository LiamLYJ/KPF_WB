import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
import seaborn as sns
from utils import *
from scipy.misc import imsave
import os

def save_ori(json_fn, ori_dir, save_fn, gain = None):
    ori_img, ori_img_gt = get_original(ori_dir, json_fn, with_gt = True)

    ori_img_gt = apply_gamma(ori_img_gt)

    imsave(save_fn + 'ori.png', ori_img)
    imsave(save_fn + 'ori_gt.png', ori_img_gt)
    if gain is not None:
        global_ori = apply_gain(ori_img, gain)
        global_ori = apply_gamma(global_ori)
        imsave(save_fn + 'global.png', global_ori)

def vis_kernel():
    fn0 = './dump_sony_64_train_ms/102MSDCF_DSC00033_108MSDCF_DSC07312_filt_0.png'
    fn1 = './dump_sony_64_train_ms/102MSDCF_DSC00033_108MSDCF_DSC07312_filt_1.png'
    fn2 = './dump_sony_64_train_ms/102MSDCF_DSC00033_108MSDCF_DSC07312_filt_2.png'
    fn3 = './dump_sony_64_train_ms/102MSDCF_DSC00033_108MSDCF_DSC07312_filt_3.png'
    fn4 = './dump_sony_64_train_ms/102MSDCF_DSC00033_108MSDCF_DSC07312_filt_4.png'
    fn5 = './dump_sony_64_train_ms/102MSDCF_DSC00033_108MSDCF_DSC07312_filt_5.png'
    fn6 = './dump_sony_64_train_ms/102MSDCF_DSC00033_108MSDCF_DSC07312_filt_6.png'
    fn7 = './dump_sony_64_train_ms/102MSDCF_DSC00033_108MSDCF_DSC07312_filt_7.png'
    fn8 = './dump_sony_64_train_ms/102MSDCF_DSC00033_108MSDCF_DSC07312_filt_8.png'

    fn = [fn0, fn1, fn2, fn3, fn4, fn5, fn6, fn7, fn8]
    img = [imread(i)/ 255.0 for i in fn]

    fig, axn = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])

    def f(x):
        return {
            0 : 'R-R',
            1 : 'R-G',
            2 : 'R-B',
            3 : 'G-R',
            4 : 'G-G',
            5 : 'G-B',
            6 : 'B-R',
            7 : 'B-G',
            8 : 'B-B',
        }[x]
    for i, ax in enumerate(axn.flat):
        sns.heatmap(img[i], ax=ax,
                cbar=i == 0,
                vmin=0, vmax=1,
                xticklabels= False,
                yticklabels= False,
                cmap = 'YlGnBu',
                cbar_ax=None if i else cbar_ax,
                square = True)
        ax.set_xlabel(f(i))
        ax.set(adjustable='box-forced', aspect='equal')
    plt.show()

def cut_and_save(fn, data_dir):
    img = imread(os.path.join(data_dir,fn))
    h,w,_ = img.shape
    img_ref = img[:, h:2*h,:]
    # img_ref = apply_gamma(img_ref)
    imsave(fn[:-4]+'_ref.png', img_ref)
    try:
        fn_np = os.path.join(data_dir, fn[:-4] + '.npy')
        np_matrix = np.load(fn_np)
        return solve_gain(np_matrix[:h, :,:], np_matrix[h:,:,:])
    except:
        return solve_gain(img[:,:h,:], img_ref)

def apply_gamma(img, gamma = 2.2):
    return np.power(img, 1.0/ gamma)


def draw_single(ori_dir, fn, label):
    img_input = imread(os.path.join(ori_dir, fn))
    img_global = apply_gain(img_input, global_gain)
    img_gt = apply_gain(img_input, label)
    img_global = apply_gamma(img_global)
    img_gt = apply_gamma(img_gt)
    imsave('./workspace/img_gt.png', img_gt)
    imsave('./workspace/img_global.png', img_global)


if __name__ == '__main__':

    # vis_kernel()
    # fn = '107MSDCF_DSC04083_109MSDCF_DSC06562.png'
    # fn = '111MSDCF_DSC07977_111MSDCF_DSC07977.png'
    # fn = '109MSDCF_DSC06534_111MSDCF_DSC07851.png'
    # fn = '108MSDCF_DSC07506_111MSDCF_DSC07918.png'
    fn = '105MSDCF_DSC03577_103MSDCF_DSC00361.png'
    # fn = '100MSDCF_DSC09972.png'
    # fn = '613_465.png'
    # fn = '597_1132.png'

    # fn = '533_1162.png'

    img_tmp = imread(fn[:-4] + '_big.png')
    img_tmp = apply_gamma(img_tmp)
    imsave(os.path.join('./workspace',fn[:-4] + '_big.png'), img_tmp)
    img_tmp = imread(fn[:-4] + '_big_pure.png')
    img_tmp = apply_gamma(img_tmp)
    imsave(os.path.join('./workspace',fn[:-4] + '_big_pure.png'), img_tmp)
    ori_dir = '../gogogo/sony_1024'
    # ori_dir = '../gogogo/cube_1024'
    data_dir = './dump_sony_train_ms'
    # data_dir = './dump_sony_train'
    # data_dir = './dump_cube_train_ms'
    # data_dir = './dump_cube_test_ms'
    json_fn = os.path.join(data_dir, fn[:-4] + '.json')
    global_gain = cut_and_save(fn, data_dir)

    save_ori(json_fn, ori_dir, './workspace/%s'%(fn[:-4]), global_gain)
