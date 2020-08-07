import os

#设置新文件名
i = 0

for i in range(2):
    for j in range(40, 50):
        oldname= './data/data_test/' + str(i) + '-' + str(j) + '_test.png'
        newname= './data/data_test/' + str(i) + '-' + str(j-40) + '_test.png'
        print(oldname, '======>', newname)

        if os.path.exists(oldname):
            # 用os模块中的rename方法对文件改名
            os.rename(oldname, newname)
            print(oldname, '======>', newname)
        else:
            j += 1


