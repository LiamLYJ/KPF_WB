import os
from glob import glob
import cv2
import scipy.io
import numpy as np
import re

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    tmp = [ tryint(c) for c in re.split('([0-9]+)', s) ]
    if tmp[0] == '-':
        tmp[1] = -1 * tmp[1]
    return tmp[1]

def sort_nicely(l):
    l.sort(key=alphanum_key)
    return l


def convert_to_8bit(arr, clip_percentile):
    arr = np.clip(arr * (255.0 / np.percentile(arr, 100 - clip_percentile, keepdims= True)), 0, 255.0)
    return arr.astype(np.uint8)

def check_cube(img_list, gt_file):

    with open(gt_file) as f:
        lines = f.readlines()

    for index, img_name in enumerate(img_list):
        #r, g, b,
        img_cv2 = cv2.imread(img_name, cv2.IMREAD_UNCHANGED).astype(np.float32)
        # line format '0.16238344 0.45326359 0.38435297\n'
        gain = lines[index][:-1]
        r_gain, g_gain, b_gain = [float(item) for item in gain.split(' ')]

        Gain_R = g_gain / r_gain
        Gain_G = g_gain / g_gain
        Gain_B = g_gain / b_gain

        processed_raw = np.maximum(img_cv2 -2048, [0,0,0])
        img12 = (processed_raw / (2**16 -1)) * 100.0

        image = convert_to_8bit(img12, 2.5)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image[:,:,0] = np.minimum(image[:,:,0] * Gain_R, 255)
        image[:,:,1] = np.minimum(image[:,:,1] * Gain_G, 255)
        image[:,:,2] = np.minimum(image[:,:,2] * Gain_B, 255)

        gamma = 1 / 2.2
        image = (image / 255.0) ** gamma * 255.0
        image = np.array(image, dtype = np.uint8)
        img8 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # print ('index: ', index + 1)
        print ('img_name : ', img_name)
        print ('Gain_R:', Gain_R)
        print ('Gain_G:', Gain_G)
        print ('Gain_B:', Gain_B)
        #
        # cv2.imshow('12bit', img12)
        cv2.imshow('8bit', img8)

        cv2.waitKey()


def make_txt_file(data_path, gt_file, save_dir = None):
    img_list = glob(os.path.join(data_path, '*.png'))
    all_num = len(img_list)
    with open(gt_file) as fp:
        gt_lines = fp.readlines()

    assert len(gt_lines) == all_num

    val_num = int(0.2 * all_num)
    filename_train = './data_txt_file/cube_train.txt'
    filename_val = './data_txt_file/cube_val.txt'

    data_dict = set()
    for index, gain in enumerate(gt_lines):
        data_dict.add((index+1, gain))

    count = 0
    with open(filename_val, 'w') as val_file:
        with open(filename_train, 'w') as train_file:
            for item in data_dict:
                item_name = os.path.join(data_path, '%d.png'%(item[0]))
                img_cv2 = cv2.imread(item_name, cv2.IMREAD_UNCHANGED).astype(np.float32)

                raw = np.maximum(img_cv2 -2048, [0,0,0])

                if save_dir is not None:
                    img8 = (np.clip(raw / raw.max(), 0, 1) * 255.0).astype(np.uint8)
                    image = cv2.resize(img8, (512,512))
                    save_name = os.path.join(save_dir,'%d.png'%(item[0]))
                    cv2.imwrite(save_name, image)

                gain = item[1]

                r_gain, g_gain, b_gain = [float(i) for i in gain.split(' ')]

                Gain_R = g_gain / r_gain
                Gain_G = g_gain / g_gain
                Gain_B = g_gain / b_gain

                gain_norm = np.linalg.norm(np.array([Gain_R, Gain_G, Gain_B]))
                Gain_R = Gain_R / gain_norm
                Gain_G = Gain_G / gain_norm
                Gain_B = Gain_B / gain_norm

                if count < val_num:
                    count += 1
                    write_file = val_file
                else:
                    write_file = train_file
                write_file.write(item_name.split('/')[-1] + '\n')
                write_file.write(str(Gain_R) + ',' + str(Gain_G) + ',' + str(Gain_B) + '\n')

                print ('index: ', item[0])
                # print ('Gain_R:', Gain_R)
                # print ('Gain_G:', Gain_G)
                # print ('Gain_B:', Gain_B)

                # raise

def make_img(img_list, save_dir):
    all_num = len(img_list)
    for item_name in img_list:
        img_cv2 = cv2.imread(item_name, cv2.IMREAD_UNCHANGED).astype(np.float32)

        raw = np.maximum(img_cv2 - 2048, [0,0,0])
        img8 = (np.clip(raw / raw.max(), 0, 1) * 255.0).astype(np.uint8)

        image = cv2.resize(img8, (512,512))
        save_name = os.path.join(save_dir,item_name.split('/')[-1])
        cv2.imwrite(save_name, image)

if __name__ == '__main__':
    data_path = '/Users/liuyongjie/Desktop/gogogo/cube_original'
    # file_gt = '/Users/lyj/cube/cube_gt.txt'
    img_list = glob(os.path.join(data_path, '*.png'))
    img_list = sort_nicely(img_list)
    # check_cube(data_path, file_gt)
    # make_txt_file(data_path, file_gt, save_dir = '/Users/lyj/Desktop/cube_')
    make_img(img_list,'/Users/liuyongjie/Desktop/gogogo/cube_1024')
