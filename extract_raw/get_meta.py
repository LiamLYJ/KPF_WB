import csv
import os
from glob import glob
file_extend = "*.ARW"
folder = '/home/cpjp/lyj/Downloads/20171128_FFCC_database'
# folder = '/Users/lyj/Desktop/check_folder_all_images/'
save_folder = '/home/cpjp/lyj/Downloads/Sony'
# save_folder = '/Users/lyj/Desktop/check_folder_all_images/txt_folder'

count = 0
all_num = 20000
for root, dirs, files in os.walk(folder):
    for dir in dirs:
        file_folder = os.path.join(root, dir)
        file_list = [os.path.basename(x) for x in glob(os.path.join(file_folder, file_extend))]
        for file in file_list:
            count += 1
            # if count > 1000:
            #     raise
            print ("read %s"%file)
            file_name = os.path.join(file_folder, file)
            file_txt = os.path.join(save_folder, file[:-4]+'.txt')
            command = 'exiftool %s > %s'%(file_name, file_txt)
            os.system(command)
            command = 'dcraw -4 -D -T %s'%(file_name)
            os.system(command)
            command = 'mv %s %s'%(file_name[:-3] + 'tiff', os.path.join(save_folder, dir + '_' + file[:-3]+'tiff'))
            os.system(command)

            WB_RBG_level = []
            Black_level = []
            White_level = []
            Orientation = []

            with open(file_txt, 'r') as txt_file:
                for line in txt_file:
                    if 'Camera Orientation' in line:
                        if ('90') in line:
                            Orientation.append([1])
                        elif ('270') in line:
                            Orientation.append([2])
                        else:
                            Orientation.append([0])
                    if 'Black Level' in line:
                        tmp = line.split()[4]
                        Black_level.append([tmp])
                    if 'White Level' in line:
                        tmp = line.split()[4]
                        White_level.append([tmp])
                    # if 'WB RGB Levels    ' in line:
                    #     tmp1 = line.split()[4]
                    #     tmp2 = line.split()[5]
                    #     tmp3 = line.split()[6]
                    #     WB_RBG_level.append([tmp1])
                    #     WB_RBG_level.append([tmp2])
                    #     WB_RBG_level.append([tmp3])
                    if 'WB RGGB Levels    ' in line:
                        tmp1 = line.split()[4]
                        tmp2 = line.split()[5]
                        tmp3 = line.split()[7]
                        WB_RBG_level.append([tmp1])
                        WB_RBG_level.append([tmp2])
                        WB_RBG_level.append([tmp3])
            command = "rm %s"%(file_txt)
            os.system(command)

            file_csv = os.path.join(save_folder, dir + '_'+ file[:-4]+'_gt.csv')
            with open (file_csv, 'w') as csv_file:
                my_write = csv.writer(csv_file)
                my_write.writerows(WB_RBG_level)
                my_write.writerows(Orientation)
                my_write.writerows(White_level)
                my_write.writerows(Black_level)
