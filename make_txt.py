import os

file_name = 'file_test.txt'

folder = '/Users/lyj/Desktop/check_rggb/check_4'
with open(file_name, 'w') as out_file:
    for root, dirs, files in os.walk(folder):
        # print (root)
        # print (dirs)
        # print (files)
        for file in files:
            if file.endswith('txt'):
                count = 0
                with open(os.path.join(root, file), 'r') as fp:
                    for line in fp:
                        if count %3 == 0:
                            out_file.write(line)
                        elif count %3 == 1:
                            out_file.write(line[:-2] + '\n')
                        count += 1
#
#folder = '/Users/lyj/Desktop/check_rggb/check_1'
#with open(file_name, 'a') as out_file:
#    for root, dirs, files in os.walk(folder):
#        # print (root)
#        # print (dirs)
#        # print (files)
#        for file in files:
#            if file.endswith('txt'):
#                count = 0
#                with open(os.path.join(root, file), 'r') as fp:
#                    for line in fp:
#                        if count %3 == 0:
#                            out_file.write(line)
#                        elif count %3 == 1:
#                            out_file.write(line[:-2] + '\n')
#                        count += 1
#
#folder = '/Users/lyj/Desktop/check_rggb/check_2'
#with open(file_name, 'a') as out_file:
#    for root, dirs, files in os.walk(folder):
#        # print (root)
#        # print (dirs)
#        # print (files)
#        for file in files:
#            if file.endswith('txt'):
#                count = 0
#                with open(os.path.join(root, file), 'r') as fp:
#                    for line in fp:
#                        if count %3 == 0:
#                            out_file.write(line)
#                        elif count %3 == 1:
#                            out_file.write(line[:-2] + '\n')
#                        count += 1
