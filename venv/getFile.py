import os
file_path = './data/data_train'
i = 0
# create files
# for i  in range(10):
#     for j in range(10):
#         file = open('./data/' + str(i) + '-' + str(j) + '.txt', 'a')
# 生成文件名txt
def file_name(path):
    for root, dirs, files in os.walk(path):
        print(root)
        print(dirs)
        print(files)

        for i in range(len(files)):
            label, num = files[i].split('-')
            f = open('./data/image_train.txt', 'a')
            f.write(str(files[i])+' ' + label + '\n')


file_name(file_path)
