from PIL import Image
import numpy as np
import sys
import os
import pdb
import matplotlib.pyplot as plt
from glob import glob
import csv

folder = '/home/cpjp/lyj/Downloads/Sony'
# folder = '/Users/lyj/Desktop/check_folder_all_images/txt_folder'
save_folder = '/home/cpjp/lyj/Downloads/Sony/preprocessed/'
# save_folder = '/Users/lyj/Desktop/check_folder_all_images/txt_folder'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)
file_extend = '*.tiff'
num_item = 20000

file_list = [os.path.basename(x) for x in glob(os.path.join(folder, file_extend))]
item = 0

file_list.sort()
for file in file_list:
    item = item + 1
    if item > num_item:
        print ('gathered enogh items ')
        break
    results = np.ones(6)
    print ("read %s"%file)
    file_csv = os.path.join(folder, file[:-5]+'_gt.csv')
    with open(file_csv, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for index, row in enumerate(reader):
            results[index] = float(row[0])
    WB_RGB_level = [results[0], results[1], results[2]]
    White_level = results[4]
    Black_level = results[5]
    rota = results[3]
    im = Image.open(os.path.join(folder, file))
    raw_data = np.array(im)
    if rota == 1:
        raw_data = np.rot90(raw_data)
    elif rota == 2:
        raw_data = np.rot90(raw_data, 3)
    else:
        assert rota == 0

    darkness_level = Black_level / 4.0
    saturation_level = White_level / 4.0

    raw_data = raw_data - darkness_level

    raw_data_rr = raw_data[0::2,0::2]
    raw_data_gr = raw_data[0::2,1::2]
    raw_data_gb = raw_data[1::2,0::2]
    raw_data_bb = raw_data[1::2,1::2]


    image_height, image_width = raw_data.shape
    print (image_height, image_width)
    Imagesize = (image_width, image_height)

    wbgain = [float(WB_RGB_level[0]) / float(WB_RGB_level[1]),
              float(WB_RGB_level[2]) / float(WB_RGB_level[1]) ]


    # raw_data_rr = raw_data_rr * wbgain[0]
    # raw_data_bb = raw_data_bb * wbgain[1]
    raw_data_gg = (raw_data_gr + raw_data_gb) * 0.5

    rgb_data = np.empty((image_height//2, image_width//2, 3), float)

    rgb_data[:,:,0] = raw_data_rr
    rgb_data[:,:,1] = raw_data_gg
    rgb_data[:,:,2] = raw_data_bb


    dgain = 2.0
    rgb_data = rgb_data * dgain
    #
    rgb_data = rgb_data/(saturation_level-darkness_level)
    rgb_data = np.clip(rgb_data, 0.0, 1.0)*255.0
    #
    # # gamma
    # rgb_data = np.power(rgb_data/255.0,1/2.2)*255.0
    if rota == 1:
        rgb_data = np.rot90(rgb_data, 3)
    elif rota == 2:
        rgb_data = np.rot90(rgb_data)
    else :
        assert rota == 0

    pil_img = Image.fromarray(np.uint8(rgb_data))

    pil_img.save( os.path.join(save_folder, file[:-4] + 'png'))

    file_txt = os.path.join(save_folder, file[:-4]+'txt')

    wbgain[0] = 1.0 / wbgain[0]
    wbgain[1] = 1.0 / wbgain[1]
    z_norm = np.linalg.norm(np.array([wbgain[0], 1.0, wbgain[1]]))
    r_gain = wbgain[0] / z_norm
    b_gain = wbgain[1] / z_norm

    with open(file_txt, 'w') as txt_file:
        txt_file.writelines(str(r_gain))
        txt_file.writelines('\n')
        txt_file.writelines(str(1.0 / z_norm))
        txt_file.writelines('\n')
        txt_file.writelines(str(b_gain))
