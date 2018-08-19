import os

folder = '/Users/lyj/Desktop/check_rggb/check_1'
out_file = './tmp_test.txt'

with open(out_file, 'w') as out_file:
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
