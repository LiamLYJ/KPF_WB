import os
from glob import glob
import cv2
import scipy.io
import numpy as np
import random


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
    filenames = glob(os.path.join(data_path, camera_name, '*.PNG'))
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
        image = cv2.resize(image, (512,512))
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


def make_txt_file(data_path, save_dir = None):
    camera_names = [
        'Canon1DsMkIII', 'Canon600D', 'FujifilmXM1', 'NikonD5200',
        'OlympusEPL6', 'PanasonicGX1', 'SamsungNX2000', 'SonyA57'
    ]
    bad_count = 0
    for id in range(8):
        camera_name = camera_names[id]
        ground_truth = scipy.io.loadmat(os.path.join(data_path,
                                        camera_name + '_gt.mat'))
        illums = ground_truth['groundtruth_illuminants']
        darkness_level = ground_truth['darkness_level']
        saturation_level = ground_truth['saturation_level']
        illums /= np.linalg.norm(illums, axis=1)[..., np.newaxis]
        filenames = glob(os.path.join(data_path, camera_name, '*.PNG'))
        filenames = sorted(filenames)
        filenames = list(filter(lambda f: f.split('/')[-1].startswith(camera_name), filenames))
        extras = {
            'darkness_level': darkness_level,
            'saturation_level': saturation_level
        }
        count = 0

        filename_train = './data_txt_file/NUS_train_%s.txt'%(camera_name)
        filename_val = './data_txt_file/NUS_val_%s.txt'%(camera_name)
        all_num = len(filenames)
        val_num = int( 1/ 7 * all_num)
        with open(filename_val, 'w') as val_file:
            with open(filename_train, 'w') as train_file:
                for i in range(len(filenames)):
                    fn = filenames[i]
                    illum = illums[i]

                    Gain_R = illum[1] / illum[0]
                    Gain_G = illum[1] / illum[1]
                    Gain_B = illum[1] / illum[2]

                    gain_norm = np.linalg.norm(np.array([Gain_R, Gain_G, Gain_B]))
                    Gain_R = Gain_R / gain_norm
                    Gain_G = Gain_G / gain_norm
                    Gain_B = Gain_B / gain_norm

                    try:
                        processed_raw = load_image(fn, extras['darkness_level'], extras['saturation_level'])
                    except:
                        print ('fn is:', fn)
                        bad_count += 1
                        print ('bad image found, skipping')
                        continue

                    img8 = (np.clip(processed_raw / processed_raw.max(), 0, 1) * 255.0).astype(np.uint8)
                    image = cv2.resize(img8, (512,512))

                    if save_dir is not None:
                        save_name = os.path.join(save_dir,fn.split('/')[-1])
                        cv2.imwrite(save_name, image)

                    if count < val_num and (random.random() > 0.6):
                        count += 1
                        write_file = val_file
                    else:
                        write_file = train_file
                    write_file.write(fn.split('/')[-1] + '\n')
                    write_file.write(str(Gain_R) + ',' + str(Gain_G) + ',' + str(Gain_B) + '\n')

                    # print ('index: ', index + 1)
                    print ('img_name : ', fn)
                    print ('Gain_R:', Gain_R)
                    print ('Gain_G:', Gain_G)
                    print ('Gain_B:', Gain_B)

    print ('bad_count: ', bad_count)

def process_nux_txt(path):
    filename_train = './data_txt_file/NUS_train.txt'
    filename_val = './data_txt_file/NUS_val.txt'
    all_train_txt = glob(os.path.join(path, 'NUS_train*'))
    all_val_txt = glob(os.path.join(path, 'NUS_val*'))
    with open(filename_train, 'w') as output:
        for item in all_train_txt:
            with open(item, 'r') as input:
                content_lines = input.readlines()
                for line in content_lines:
                    output.write(line)
    with open(filename_val, 'w') as output:
        for item in all_val_txt:
            with open(item, 'r') as input:
                content_lines = input.readlines()
                for line in content_lines:
                    output.write(line)


if __name__ == '__main__':
    data_path = '/Users/lyj/Documents/AWB_data/nus/'
    # check_NUS(data_path, id=0)
    # make_txt_file(data_path, save_dir = '/Users/lyj/Desktop/nus')
    path = './data_txt_file'
    process_nux_txt(path)
