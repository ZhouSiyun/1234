# coding:utf-8
import tensorflow as tf
import os
from image_process import imageProcess


image_train_path = './data/data_train/'
label_train_path = './data/image_train.txt'
tfRecord_train = './data/my_train.tfrecords'

image_test_path = './data/data_test/'
label_test_path = './data/image_test.txt'
tfRecord_test = './data/my_test.tfrecords'
data_path = './data'
resize_height = 100
resize_width = 100


def write_tfRecord(tfRecordName, image_path, label_path):  # 创建tfrecord文档
    writer = tf.python_io.TFRecordWriter(tfRecordName)
    num_pic = 0
    f = open(label_path, 'r')  # 打开的是txt文档
    contents = f.readlines()  # 按行读取
    f.close()
    for content in contents:
        value = content.split()  # 图的名字和label
        print(value)
        img_path = image_path + value[0]  # 图像的路径是图像文件夹的名字+图像的名字
        print(image_path)
        img = imageProcess(img_path)  # 对图像进行处理
        img_raw = img.tobytes()  # 将图像转换成二进制
        labels = [0] * 6
        labels[int(value[1])] = 1  # 独热转换，将标签转换成独热的形式
        # 将数据处理成二进制方面，一般是为了提升IO效率和方便管理数据。
        example = tf.train.Example(features=tf.train.Features(feature={
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
        }))
        writer.write(example.SerializeToString())
        num_pic += 1
        print("the number of picture:", num_pic)
    writer.close()
    print("write tfrecord successful")


def generate_tfRecord():
    isExists = os.path.exists(data_path)
    if not isExists:
        os.makedirs(data_path)
        print('The directory was created successfully')
    else:
        print('directory already exists')
    write_tfRecord(tfRecord_train, image_train_path, label_train_path)  # tfRecordName, image_path, label_path
    write_tfRecord(tfRecord_test, image_test_path, label_test_path)
    # 此处就是将所有的图像特征转换成TFRECOR文档的格式


def read_tfRecord(tfRecord_path):
    filename_queue = tf.train.string_input_producer([tfRecord_path], shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([6], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string)
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    # img.set_shape([100,100,1])
    img = tf.reshape(img, [100, 100, 1])
    img = tf.cast(img, tf.float32) * (1. / 255)
    label = tf.cast(features['label'], tf.float32)
    return img, label


def get_tfrecord(num, isTrain=True):
    if isTrain:
        tfRecord_path = tfRecord_train  # 训练图像的tfrecord路径
    else:
        tfRecord_path = tfRecord_test  # 测试图像的tfrecord路径
    img, label = read_tfRecord(tfRecord_path)  # 返回的是图像和标签
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=num,
                                                    num_threads=2,
                                                    capacity=100,
                                                    min_after_dequeue=20)
    return img_batch, label_batch


def main():
    generate_tfRecord()


if __name__ == '__main__':
    main()