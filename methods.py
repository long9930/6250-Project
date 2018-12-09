#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 23:16:29 2018

@author: chenming
"""


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import sklearn as sk
import tensorflow as tf
import os, random, time
from plots import plot_learning_curves

def cnn_AlexNet(x,ratio = 0.01,n_classes=2):
    def conv2d(img, w, b, k = 1):
        return tf.nn.tanh(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, k, k, 1], padding='SAME'),b))

    def max_pool(img, k):
        return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
    wc1 = tf.Variable(tf.random_normal([11, 11, 1, 32])*ratio, name="wc1")
    bc1 = tf.Variable(tf.random_normal([32])*ratio, name="bc1")
    # stride 64 x 64
    # pool 32 x 32
    wc2 = tf.Variable(tf.random_normal([3, 3, 32, 128])*ratio, name="wc2")
    bc2 = tf.Variable(tf.random_normal([128])*ratio, name="bc2")
    # pool 16 x 16
    wc3 = tf.Variable(tf.random_normal([3, 3, 128, 96])*ratio, name="wc3")
    bc3 = tf.Variable(tf.random_normal([96])*ratio, name="bc3")
    # pool 16 x 16
    wc4 = tf.Variable(tf.random_normal([3, 3, 96, 64])*ratio, name="wc4")
    bc4 = tf.Variable(tf.random_normal([64])*ratio, name="bc4")
    # pool 8x8
    wd1 = tf.Variable(tf.random_normal([8*8*64, 512])*ratio, name="wd1")
    bd1 = tf.Variable(tf.random_normal([512])*ratio, name="bd1")
    wd2 = tf.Variable(tf.random_normal([512, 256])*ratio, name="wd2")
    bd2 = tf.Variable(tf.random_normal([256])*ratio, name="bd2")
    wout = tf.Variable(tf.random_normal([256, n_classes])*ratio, name="wout")
    bout = tf.Variable(tf.random_normal([n_classes])*ratio, name="bout")

    # conv layer
    #x = tf.Print(x, [x])
    conv1 = conv2d(x,wc1,bc1, k = 4)
    conv1 = max_pool(conv1, k=2)
    # conv layer
    conv2 = conv2d(conv1,wc2,bc2)
    conv2 = max_pool(conv2, k=2)
    # conv2 = avg_pool(conv2, k=2)

    # dropout to reduce overfitting
    keep_prob = tf.placeholder(tf.float32)
    conv2 = tf.nn.dropout(conv2, keep_prob)

    # conv layer
    conv3= conv2d(conv2,wc3,bc3)
    # dropout to reduce overfitting
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # conv layer
    conv4 = conv2d(conv3,wc4,bc4)
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 64*64 matrix.
    conv4 = max_pool(conv4, k=2)

    # dropout to reduce overfitting
    conv4 = tf.nn.dropout(conv4, keep_prob)

    # fc 1
    dense1 = tf.reshape(conv4, [-1, wd1.get_shape().as_list()[0]])
    dense1 = tf.nn.tanh(tf.add(tf.matmul(dense1, wd1),bd1))
    dense1 = tf.nn.dropout(dense1, keep_prob)

    # fc 2
    dense2 = tf.reshape(dense1, [-1, wd2.get_shape().as_list()[0]])
    dense2 = tf.nn.tanh(tf.add(tf.matmul(dense2, wd2),bd2))
    dense2 = tf.nn.dropout(dense2, keep_prob)

    # prediction
    pred = tf.add(tf.matmul(dense2, wout), bout)
    pred = tf.Print(pred, [pred])
    return pred, keep_prob


def cnn_LeNet5(x,ratio = 0.01,n_classes=2):
    def conv2d(img, w, b, k = 1):
        return tf.nn.tanh(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, k, k, 1], padding='SAME'),b))
    def avg_pool(img, k):
        return tf.nn.avg_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    wc1 = tf.Variable(tf.random_normal([11, 11, 1, 16])*ratio, name="wc1")
    bc1 = tf.Variable(tf.random_normal([16])*ratio, name="bc1")
    # stride 64 x 64
    # pool 32 x 32
    wc2 = tf.Variable(tf.random_normal([11, 11, 16, 32])*ratio, name="wc2")
    bc2 = tf.Variable(tf.random_normal([32])*ratio, name="bc2")
    # pool 16 x 16

    wd1 = tf.Variable(tf.random_normal([8*8*32, 256])*ratio, name="wd1")
    bd1 = tf.Variable(tf.random_normal([256])*ratio, name="bd1")
    wd2 = tf.Variable(tf.random_normal([256, 128])*ratio, name="wd2")
    bd2 = tf.Variable(tf.random_normal([128])*ratio, name="bd2")
    wout = tf.Variable(tf.random_normal([128, n_classes])*ratio, name="wout")
    bout = tf.Variable(tf.random_normal([n_classes])*ratio, name="bout")

    # conv layer
    #x = tf.Print(x, [x])
    conv1 = conv2d(x,wc1,bc1, k = 1)
    conv1 = avg_pool(conv1, k=2)
    
    
    # conv layer
    conv2 = conv2d(conv1,wc2,bc2,k=1)
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 64*64 matrix.
    conv2 = avg_pool(conv2, k=2)
    # conv2 = avg_pool(conv2, k=2)

    # dropout to reduce overfitting
    keep_prob = tf.placeholder(tf.float32)
    conv2 = tf.nn.dropout(conv2, keep_prob)

    # fc 1
    dense1 = tf.reshape(conv2, [-1, wd1.get_shape().as_list()[0]])
    dense1 = tf.nn.tanh(tf.add(tf.matmul(dense1, wd1),bd1))
    dense1 = tf.nn.dropout(dense1, keep_prob)

    # fc 2
    dense2 = tf.reshape(dense1, [-1, wd2.get_shape().as_list()[0]])
    dense2 = tf.nn.tanh(tf.add(tf.matmul(dense2, wd2),bd2))
    dense2 = tf.nn.dropout(dense2, keep_prob)

    # prediction
    pred = tf.add(tf.matmul(dense2, wout), bout)
    pred = tf.Print(pred, [pred])
    return pred, keep_prob


