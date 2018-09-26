import os
from glob import glob
import cv2
import scipy.io
import numpy as np

def convert_to_8bit(arr, clip_percentile):
    arr = np.clip(arr * (255.0 / np.percentile(arr, 100 - clip_percentile, keepdims= True)), 0, 255.0)
    return arr.astype(np.uint8)

def check_gehler(img_list, mat):
    for index, img_name in enumerate(img_list):
        #a, r, g, b,
        img_cv2 = cv2.imread(img_name, cv2.IMREAD_UNCHANGED).astype(np.float32)

        Gain_R = float(np.max(mat['real_rgb'][index])) / float(mat['real_rgb'][index][0])
        Gain_G = float(np.max(mat['real_rgb'][index])) / float(mat['real_rgb'][index][1])
        Gain_B = float(np.max(mat['real_rgb'][index])) / float(mat['real_rgb'][index][2])

        if 'IMG' in img_name:
            raw = np.maximum(img_cv2 - 129, [0,0,0])
            img12 = ( raw / (2**12 -1)) * 100.0
        else:
            raw = np.maximum(img_cv2 - 1, [0,0,0])
            img12 = ( raw / (2**12 -1)) * 100.0

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
        # print ('img_name : ', img_name)
        # print ('Gain_R:', Gain_R)
        # print ('Gain_G:', Gain_G)
        # print ('Gain_B:', Gain_B)
        #
        # cv2.imshow('12bit', img12)
        cv2.imshow('8bit', img8)

        cv2.waitKey()


# we found two camera cannot be trian together
# the whole dataset of gehler has 586, 82-camera0 ,remian - camera1
# so we only use the caerame1 :
def make_txt_file(img_list, mat, save_dir = None):
    all_num = len(img_list)

    val_num = 80
    filename_train = './data_txt_file/gehler_train_only_one.txt'
    filename_val = './data_txt_file/gehler_val_only_one.txt'

    data_dict = {}
    for index, item_name in enumerate(img_list):
        data_dict[item_name] = index

    count = 0
    with open(filename_val, 'w') as val_file:
        with open(filename_train, 'w') as train_file:
            for item_name in data_dict:
                img_cv2 = cv2.imread(item_name, cv2.IMREAD_UNCHANGED).astype(np.float32)

                if 'IMG' in item_name:
                    raw = np.maximum(img_cv2 - 129, [0,0,0])
                else:
                    raw = np.maximum(img_cv2 - 1, [0,0,0])

                if save_dir is not None:
                    img8 = (np.clip(raw / raw.max(), 0, 1) * 255.0).astype(np.uint8)
                    image = cv2.resize(img8, (512,512))
                    save_name = os.path.join(save_dir,item_name.split('/')[-1])
                    cv2.imwrite(save_name, image)

                index = data_dict[item_name]
                Gain_R = float(np.max(mat['real_rgb'][index])) / float(mat['real_rgb'][index][0])
                Gain_G = float(np.max(mat['real_rgb'][index])) / float(mat['real_rgb'][index][1])
                Gain_B = float(np.max(mat['real_rgb'][index])) / float(mat['real_rgb'][index][2])

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
                # print ('item_name:', item_name)
                # print ('index: ', index)
                # print ('Gain_R:', Gain_R)
                # print ('Gain_G:', Gain_G)
                # print ('Gain_B:', Gain_B)


def make_img(img_list, save_dir):
    all_num = len(img_list)
    for item_name in img_list:
        img_cv2 = cv2.imread(item_name, cv2.IMREAD_UNCHANGED).astype(np.float32)

        if 'IMG' in item_name:
            raw = np.maximum(img_cv2 - 129, [0,0,0])
        else:
            raw = np.maximum(img_cv2 - 1, [0,0,0])

        # img16 = (np.clip(raw / raw.max(), 0, 1) * 65535.0).astype(np.uint16)
        img8 = (np.clip(raw / raw.max(), 0, 1) * 255.0).astype(np.uint8)

        image = cv2.resize(img8, (1024,1024))
        save_name = os.path.join(save_dir,item_name.split('/')[-1])
        cv2.imwrite(save_name, image)


if __name__ == '__main__':
    data_path = '/Users/liuyongjie/Desktop/gogogo/gehler_original'
    img_list = glob(os.path.join(data_path, '*.png'))
    img_list = sorted(img_list)

    # mat = (scipy.io.loadmat('/Users/lyj/Documents/AWB_data/gehler_original/real_illum_568.mat', squeeze_me = True, struct_as_record = False))
    # make_txt_file(img_list, mat, '/Users/lyj/Desktop/gehler')
    make_img(img_list, '/Users/liuyongjie/Desktop/gogogo/gehler_1024')
    # check_gehler(img_list, mat)
