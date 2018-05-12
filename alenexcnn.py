import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import loaddata
import _pickle as pickle
import seaborn

class CnnChaoCan(object):
# 模型参数
    learning_rate = 0.00001
    training_iters =40
    batch_size = 70
    display_step = 5
    n_features =37632  #112*112*3
    n_classes = 480
    n_fc1 = 384
    n_fc2 = 4096

class faceCNN(object):
    def __init__(self,config):
    # 构建模型
        self.config=config
        self.x = tf.placeholder(tf.float32, [None, 112,112,3])
        self.y = tf.placeholder(tf.float32, [None, config.n_classes])
        self.keep_prob=tf.placeholder(tf.float32)
        self.w = {
            'conv1': tf.Variable(tf.truncated_normal([5, 5, 3, 64], stddev=5e-2)),
            'conv2': tf.Variable(tf.truncated_normal([5, 5, 64, 64], stddev=0.1)),
            'fc1': tf.Variable(tf.truncated_normal([28*28* 64, config.n_fc1], stddev=0.04)),
            'fc2': tf.Variable(tf.truncated_normal([config.n_fc1, config.n_fc2], stddev=0.1)),
            'fc3': tf.Variable(tf.truncated_normal([config.n_fc2, config.n_classes], stddev=1 / 192.0))
        }
        self.b = {
            'conv1': tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[64])),
            'conv2': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[64])),
            'fc1': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[config.n_fc1])),
            'fc2': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[config.n_fc2])),
            'fc3': tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[config.n_classes]))
        }
        self.cnn()
    def cnn(self):
        #x4d = tf.reshape(x, [-1, 224,224, 3])
        # 卷积层 1
        conv1 = tf.nn.conv2d(self.x, self.w['conv1'], strides=[1, 1, 1, 1], padding='SAME')
        conv1 = tf.nn.bias_add(conv1, self.b['conv1'])
        conv1 = tf.nn.relu(conv1)
        # 池化层 1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        # LRN层，Local Response Normalization
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        # 卷积层 2
        conv2 = tf.nn.conv2d(norm1, self.w['conv2'], strides=[1, 1, 1, 1], padding='SAME')
        conv2 = tf.nn.bias_add(conv2, self.b['conv2'])
        conv2 = tf.nn.relu(conv2)
         # LRN层，Local Response Normalization
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
         # 池化层 2
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        reshape = tf.reshape(pool2, [-1, 28* 28* 64])
        dim = reshape.get_shape()[1].value
        fc1 = tf.add(tf.matmul(reshape, self.w['fc1']), self.b['fc1'])
        fc1 = tf.nn.relu(fc1)
        # 全连接层 2
        fc2 = tf.add(tf.matmul(fc1, self.w['fc2']), self.b['fc2'])
        fc2 = tf.nn.relu(fc2)
        # 全连接层 3, 即分类层
        fc2=tf.nn.dropout(fc2,self.keep_prob)
        self.fc3 = tf.add(tf.matmul(fc2, self.w['fc3']), self.b['fc3'])
        self.softmax=tf.nn.softmax(self.fc3,name="prob")
        # 定义损失
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.fc3, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.cost)
        # 评估模型
        self.correct_pred = tf.equal(tf.argmax(self.fc3, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


