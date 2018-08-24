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
        img_cv2 = cv2.imread(os.path.join(data_path, img_name), cv2.IMREAD_UNCHANGED).astype(np.float32)
        Gain_R = float(np.max(mat['real+rgb'][index])) / float(mat['real_rgb'][index][0])
        Gain_G = float(np.max(mat['real+rgb'][index])) / float(mat['real_rgb'][index][1])
        Gain_B = float(np.max(mat['real+rgb'][index])) / float(mat['real_rgb'][index][2])


        img16 = (img_cv2 / (2**16 -1)) * 100.0
        image = convert_to_8bit(img16, 2.5)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image[:,:,0] = np.minimum(image[:,:,0] * Gain_R, 255)
        image[:,:,1] = np.minimum(image[:,:,1] * Gain_R, 255)
        image[:,:,2] = np.minimum(image[:,:,2] * Gain_R, 255)

        gamma = 1 / 2.2
        image = (image / 255.0) ** gamma * 255.0
        image = np.array(image, dtype = np.uint8)
        img8 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        cv2.imshow('16bit', img16)
        cv2.imshow('8bit', img8)

        cv2.waitKey()

def make_txt_file(img_list, mat, save_dir):
    all_num = len(img_list)
    train_num = int(0.75 * all_num)
    val_num = all_num - train_num
    filename_train = 'gehler_train.txt'
    filename_val = 'gehler_val.txt'

    data_dict = {}
    for index, item_name in enumerate(img_list):
        data_dict[item_name] = index

    with open(filename_val, 'w') as val_file:
        with open(filename_train, 'w') as train_file:
            for item_name in data_dict:
                img_cv2 = cv2.imread(item_name, cv2.IMREAD_UNCHANGED).astype(np.float32)
                img16 = (img_cv2 / (2**16 -1)) * 100.0
                image = convert_to_8bit(img16, 2.5)
                save_name = os.path.join(save_dir,item_name.split('/')[-1])
                cv2.imwrite(save_name, image)

                index = data_dict[item_name]
                Gain_R = float(np.max(mat['real_rgb'][index])) / float(mat['real_rgb'][index][0])
                Gain_G = float(np.max(mat['real_rgb'][index])) / float(mat['real_rgb'][index][1])
                Gain_B = float(np.max(mat['real_rgb'][index])) / float(mat['real_rgb'][index][2])
                if index < val_num:
                    write_file = val_file
                else:
                    write_file = train_file
                write_file.write(item_name.split('/')[-1] + '\n')
                write_file.write(str(Gain_R) + ',' + str(Gain_G) + ',' + str(Gain_B) + '\n')


if __name__ == '__main__':
    data_path = 'png'
    img_list = glob(os.path.join(data_path, '*.png'))
    img_list = sorted(img_list)
    mat = (scipy.io.loadmat('real_illum_568.mat', squeeze_me = True, struct_as_record = False))
    make_txt_file(img_list, mat, 'wheretosave')
