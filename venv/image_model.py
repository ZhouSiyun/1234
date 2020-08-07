
#coding:utf-8
import tensorflow as tf
from venv.gesture_generate import get_tfrecord

IMAGE_SIZE = 100
NUM_CHANNELS = 1
CONV1_SIZE = 4
CONV1_KERNEL_NUM = 8
CONV2_SIZE = 2
CONV2_KERNEL_NUM = 16
FC_SIZE = 512
OUTPUT_NODE = 6

BATCH_SIZE = 60
LEARNING_RATE_BASE =  0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="./model/"
MODEL_NAME="my_geature_model"
train_num_examples = 1080#2


def get_weight(shape, regularizer):
	w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
	if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    # 把变量放入一个集合，把很多变量变成一个列表
	return w

def get_bias(shape):
	b = tf.Variable(tf.zeros(shape))
	return b

def conv2d(x,w):
	return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_8x8(x):
	return tf.nn.max_pool(x, ksize=[1, 8, 8, 1], strides=[1, 4, 4, 1], padding='SAME')

def max_pool_4x4(x):
	return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='SAME')

'''
x: 图像（的占位符）
train：是bool型，在test和app中为False，在backward中为True
'''
def model(x, train, regularizer):
    # 进行两层的卷积操作
    conv1_w = get_weight([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM], regularizer)
    conv1_b = get_bias([CONV1_KERNEL_NUM])
    conv1 = conv2d(x, conv1_w)
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    pool1 = max_pool_8x8(relu1)

    conv2_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM],regularizer)
    conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv2 = conv2d(pool1, conv2_w)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    pool2 = max_pool_4x4(relu2)

    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    fc1_w = get_weight([nodes, FC_SIZE], regularizer)
    fc1_b = get_bias([FC_SIZE])
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
    if train: fc1 = tf.nn.dropout(fc1, 0.5)

    fc2_w = get_weight([FC_SIZE, OUTPUT_NODE], regularizer)
    fc2_b = get_bias([OUTPUT_NODE])
    y = tf.matmul(fc1, fc2_w) + fc2_b   # softmax函数 也就是CNN的最后一层
    return y   # 返回值用来确定输入图像的类别

X = tf.placeholder(tf.float32,[
	BATCH_SIZE,
	IMAGE_SIZE,
	IMAGE_SIZE,
	NUM_CHANNELS])
Y = tf.placeholder(tf.float32, [None, OUTPUT_NODE])

y = model(X, True, REGULARIZER)
global_step = tf.Variable(0, trainable=False)
Y_ = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(Y, 1))
cost = tf.reduce_mean(Y_) # 损失函数
loss = cost + tf.add_n(tf.get_collection('losses'))
    # 指数衰减学习率
# 1. 先使用一个较大的学习率，得出一个最优解
# 2. 通过迭代逐步减少学习率，使模型的训练后期更加稳定
learning_rate = tf.train.exponential_decay(
    LEARNING_RATE_BASE,
    global_step,
    train_num_examples / BATCH_SIZE,
    LEARNING_RATE_DECAY,
    staircase=True)
# 梯度下降法（书中使用的是RMSPropOptimizer）
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
# 滑动平均操作
# 这一步存在的意义就是提高模型在测试数据上的健壮性
ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
ema_op = ema.apply(tf.trainable_variables())
with tf.control_dependencies([train_step, ema_op]):
    train_op = tf.no_op(name='train')

# 数据输入的部分
img_batch, label_batch = gesture_generate.get_tfrecord(BATCH_SIZE, isTrain=True)

#venv/gesture_generate.py

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    coord = tf.train.Coordinator()  # 4
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 5

    for i in range(STEPS):
        xs, ys = sess.run([img_batch, label_batch])  # 6
        _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
        if i % 100 == 0:
            print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
            saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

    coord.request_stop()  # 7
    coord.join(threads)  # 8



# 保存模型
saver = tf.train.Saver()
save_path = saver.save(sess, "softmax_mnist")
print("Model save path: %s" % save_path)

