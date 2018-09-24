import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
import seaborn as sns
from utils import *
from scipy.misc import imsave
import os
import re

def save_ori(json_fn, ori_dir, save_fn, gain = None):
    ori_img, ori_img_gt, _, _ = get_original(ori_dir, json_fn, with_gt = True)

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


def draw_single(ori_dir, data_dir, fn):
    log_file = os.path.join(data_dir, 'log.txt')
    with open(log_file, 'r') as f:
        contents = f.readlines()
    for i in range(len(contents)):
        if fn in contents[i]:
            est = re.findall(r"[-+]?\d*\.\d+|\d+", contents[i+1])
            gt = re.findall(r"[-+]?\d*\.\d+|\d+", contents[i+2])
            error = re.findall(r"[-+]?\d*\.\d+|\d+", contents[i+3])
            est = [float(item) for item in est]
            gt = [float(item) for item in gt]
            error = [float(item) for item in error]
            # print (est, gt, error)
    img_ori = imread(os.path.join(ori_dir, fn))
    img_gt = apply_gain(img_ori, gt)
    img_gt = apply_gamma(img_gt)
    img_est = apply_gain(img_ori, est)
    img_est = apply_gamma(img_est)
    imsave('./work_space_all/%s_gt.png'%(fn[:-4]), img_gt)
    imsave('./work_space_all/%s_%f.png'%(fn[:-4] + '_est', error[0]), img_est)
    imsave('./work_space_all/%s_input.png'%(fn[:-4]), img_ori)


def draw_multi(ori_dir, data_dir, fn):
    file_np = os.path.join(data_dir, fn + '.npy')
    tmp = np.load(file_np)

    size = tmp.shape[1]

    img0 = tmp[:size,:,:]
    img2 = tmp[size:,:,:]

    img2 = np.clip(img2, 0,500)
    print ('min ref:', np.min(img2))
    print ('max ref:', np.max(img2))

    json_file = os.path.join(data_dir, fn + '.json')

    concat_original, concat_gt, scale_h, scale_w = get_original(data_dir= ori_dir, json_file = json_file, with_gt = True)
    save_path = './work_space_all_ms'

    with open(json_file, 'r') as json_f:
        label1_2 = json.load(json_f)['label1_2']

    gain_box, clus_img, clus_labels = gain_fitting(img0, img2, is_local = True, n_clusters =2, gamma = 2.0, with_clus = True)
    errors = gain_fitting_mask(img0, img2, clus_labels, gt_label = label1_2 )
    print (errors)

    imsave(os.path.join(save_path, fn+'_clus.png'), clus_img)
    big = apply_gain_box(concat_original, gain_box, scale_h, scale_w)
    big = apply_gamma(big)
    concat_gt = apply_gamma(concat_gt)
    imsave(os.path.join(save_path, fn + '_est_%f_%f.png'%(errors[0], errors[1])), big)
    imsave(os.path.join(save_path, fn + '_gt.png'), concat_gt)
    imsave(os.path.join(save_path, fn + '_input.png'), concat_original)


if __name__ == '__main__':

    ######### draw multi #######################
    # fn = ['100MSDCF_DSC09744_100MSDCF_DSC09775',  # big confi, small error
    #     '100MSDCF_DSC09998_100MSDCF_DSC09930', '103MSDCF_DSC00681_103MSDCF_DSC00831', # big confi, big error
    #     '104MSDCF_DSC03862_104MSDCF_DSC03812',  # small confi,  small error
    #     '103MSDCF_DSC00681_104MSDCF_DSC03800'] # small confi, big error

    # ori_dir = './data/sony_1024'
    # data_dir = './dump_sony_test_ms'
    #

    # fn = [
    #     # '237_319',  # big confi, small error
    # #     '865_575', # big confi, big error
    # #     '25_21',  # small confi,  small error
    #     '277_934' # small confi, big error
    #     ]
    # ori_dir = './data/cube_1024'
    # data_dir = './dump_cube_test_ms'
    #

    # fn = [
    #     # '',  # big confi, small error
        # 'FujifilmXM1_0016_OlympusEPL6_0006', # big confi, big error
        # '',  # small confi,  small error
        # 'SonyA57_0048_SonyA57_0033' # small confi, big error
        # ]
    # fn = ['PanasonicGX1_0189_PanasonicGX1_0160']
    # fn = ['FujifilmXM1_0058_FujifilmXM1_0047']
    # ori_dir = './data/nus_1024'
    # data_dir = './dump_nus_test_ms'
    # data_dir = './dump_nus_train_ms'


    fn = [
        # '',  # big confi, small error
        'IMG_0845_IMG_0378', # big confi, big error
        'IMG_0284_IMG_0598',  # small confi,  small error
        # 'IMG_0424_IMG_0870' # small confi, big error
        ]
    ori_dir = './data/gehler_1024'
    data_dir = './dump_gehler_test_ms'
    # data_dir = './dump_gehler_train_ms'
    # fn = ['IMG_0480_IMG_0621']


    for item in fn:
        draw_multi(ori_dir, data_dir, item)

    raise
    ###############################################################


    ##### draw single ##########################
    # fn = ['100MSDCF_DSC09722.png', '100MSDCF_DSC09748.png',  # big confi, small error
    #     '104MSDCF_DSC03862.png', '103MSDCF_DSC00532.png' , '103MSDCF_DSC00831.png', # big confi, big error
    #     '104MSDCF_DSC03856.png', '104MSDCF_DSC03914.png',  # small confi,  small error
    #     '104MSDCF_DSC03800.png',  '104MSDCF_DSC03812.png',] # small confi, big error
    #
    # ori_dir = './data/sony_1024'
    # data_dir = './dump_sony_test'

    # fn = [ '819.png',  # big confi, small error
    #      '865.png', # big confi, big error
    #      '746.png',  # small confi,  small error
    #      '934.png',] # small confi, big error
    #
    # ori_dir = './data/cube_1024'
    # data_dir = './dump_cube_test'

    # fn = [ 'SamsungNX2000_0010.PNG',  # big confi, small error
    #      'Canon1DsMkIII_0001.PNG', # big confi, big error
    #      'Canon1DsMkIII_0016.PNG',  # small confi,  small error
    #      'SamsungNX2000_0006.PNG',] # small confi, big error
    #
    # ori_dir = './data/nus_1024'
    # data_dir = './dump_nus_test'

    # fn = [ 'IMG_0307.png',  # big confi, small error
    #      'IMG_0324.png', # big confi, big error
    #      'IMG_0370.png',  # small confi,  small error
    #      'IMG_0298.png',] # small confi, big error
    #
    # ori_dir = './data/gehler_1024'
    # data_dir = './dump_gehler_test'
    #
    # for item in fn:
    #     draw_single(ori_dir, data_dir, item)
    # raise
    #############################################

    img_tmp = imread(fn[:-4] + '_big.png')
    img_tmp = apply_gamma(img_tmp)
    imsave(os.path.join('./workspace',fn[:-4] + '_big.png'), img_tmp)
    img_tmp = imread(fn[:-4] + '_big_pure.png')
    img_tmp = apply_gamma(img_tmp)
    imsave(os.path.join('./workspace',fn[:-4] + '_big_pure.png'), img_tmp)
    json_fn = os.path.join(data_dir, fn[:-4] + '.json')
    global_gain = cut_and_save(fn, data_dir)

    save_ori(json_fn, ori_dir, './workspace/%s'%(fn[:-4]), global_gain)
