import os
from glob import glob
import cv2
import scipy.io
import numpy as np


def load_image(fn, darkness_level, saturation_level):
    raw = cv2.imread(fn, cv2.IMREAD_UNCHANGED).astype(np.float32)
    raw = np.maximum(raw - darkness_level, [0, 0, 0])
    raw *= 1.0 / saturation_level
    return raw

def convert_to_8bit(arr, clip_percentile):
    arr = np.clip(arr * (255.0 / np.percentile(arr, 100 - clip_percentile, keepdims= True)), 0, 255.0)
    return arr.astype(np.uint8)


def check_NUS(data_path, id = None):
    camera_names = [
        'Canon1DsMkIII', 'Canon600D', 'FujifilmXM1', 'NikonD5200',
        'OlympusEPL6', 'PanasonicGX1', 'SamsungNX2000', 'SonyA57'
    ]

    camera_name = camera_names[id]
    ground_truth = scipy.io.loadmat(os.path.join(data_path,
                                    camera_name + '_gt.mat'))
    illums = ground_truth['groundtruth_illuminants']
    darkness_level = ground_truth['darkness_level']
    saturation_level = ground_truth['saturation_level']
    illums /= np.linalg.norm(illums, axis=1)[..., np.newaxis]
    filenames = glob(os.path.join(data_path, '*.PNG'))
    filenames = sorted(filenames)
    filenames = list(filter(lambda f: f.split('/')[-1].startswith(camera_name), filenames))
    extras = {
        'darkness_level': darkness_level,
        'saturation_level': saturation_level
    }

    for i in range(len(filenames)):
        fn = filenames[i]
        illum = illums[i]

        Gain_R = illum[1] / illum[0]
        Gain_G = illum[1] / illum[1]
        Gain_B = illum[1] / illum[2]

        processed_raw = load_image(fn, extras['darkness_level'], extras['saturation_level'])

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
        print ('img_name : ', fn)
        print ('Gain_R:', Gain_R)
        print ('Gain_G:', Gain_G)
        print ('Gain_B:', Gain_B)
        #
        # cv2.imshow('12bit', img12)
        cv2.imshow('8bit', img8)

        cv2.waitKey()


def make_txt_file(data_path, save_dir = None, id = 0):
    camera_names = [
        'Canon1DsMkIII', 'Canon600D', 'FujifilmXM1', 'NikonD5200',
        'OlympusEPL6', 'PanasonicGX1', 'SamsungNX2000', 'SonyA57'
    ]

    camera_name = camera_names[id]
    ground_truth = scipy.io.loadmat(os.path.join(data_path,
                                    camera_name + '_gt.mat'))
    illums = ground_truth['groundtruth_illuminants']
    darkness_level = ground_truth['darkness_level']
    saturation_level = ground_truth['saturation_level']
    illums /= np.linalg.norm(illums, axis=1)[..., np.newaxis]
    filenames = glob(os.path.join(data_path, '*.PNG'))
    filenames = sorted(filenames)
    filenames = list(filter(lambda f: f.split('/')[-1].startswith(camera_name), filenames))
    extras = {
        'darkness_level': darkness_level,
        'saturation_level': saturation_level
    }
    count = 0

    train_file = './data_txt_file/NUS_train.txt'
    val_file = './data_txt_file/NUS_val.txt'
    val_num = 50
    for i in range(len(filenames)):
        fn = filenames[i]
        illum = illums[i]

        Gain_R = illum[1] / illum[0]
        Gain_G = illum[1] / illum[1]
        Gain_B = illum[1] / illum[2]

        processed_raw = load_image(fn, extras['darkness_level'], extras['saturation_level'])

        img12 = (processed_raw / (2**16 -1)) * 100.0

        image = convert_to_8bit(img12, 2.5)
        imgae = cv2.imresize(image, (512,512))

        if save_dir is not None:
            save_name = os.path.join(save_dir,item_name.split('/')[-1])
            cv2.imwrite(save_name, image)

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

        # print ('index: ', index + 1)
        print ('img_name : ', fn)
        print ('Gain_R:', Gain_R)
        print ('Gain_G:', Gain_G)
        print ('Gain_B:', Gain_B)
        #
        # cv2.imshow('12bit', img12)
        cv2.imshow('8bit', img8)

        cv2.waitKey()

if __name__ == '__main__':
    data_path = '/Users/lyj/Desktop/Canon_PNG'
    check_NUS(data_path, id=0)
