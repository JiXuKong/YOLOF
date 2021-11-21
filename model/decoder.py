import tensorflow as tf
import numpy as np
import math
from model.layers import conv2d, bn_, act

class Decoder(object):
    """
    Head Decoder for YOLOF.

    This module contains two types of components:
        - A classification head with two 3x3 convolutions and one
            classification 3x3 convolution
        - A regression head with four 3x3 convolutions, one regression 3x3
          convolution, and one implicit objectness 3x3 convolution
    """
    def __init__(self, in_channels,
        num_classes,
        num_anchors,
        cls_num_convs,
        reg_num_convs,
        prior_prob,
        is_training):

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.cls_num_convs = cls_num_convs
        self.reg_num_convs = reg_num_convs
        self.prior_prob = prior_prob
        self.is_training = is_training
        
        self.INF = 1e8

    def build_layers(self, x):
        x_cls = tf.identity(x)
        x_reg = tf.identity(x)
        for i in range(self.cls_num_convs):
            x_cls = conv2d(x_cls, self.in_channels,
                                      kernel=3,
                                      is_training=self.is_training,
                                      scope='ClsSubnetConv'+str(i))
            x_cls = bn_(x_cls, is_training=self.is_training, scope='ClsSubnetNorm'+str(i))
            x_cls = act(x_cls)

        for i in range(self.reg_num_convs):
            x_reg = conv2d(x, self.in_channels,
                                      kernel=3,
                                      is_training=self.is_training,
                                      scope='BboxSubnetConv'+str(i))
            x_reg = bn_(x_reg, is_training=self.is_training, scope='BboxSubnetNorm'+str(i))
            x_reg = act(x_reg)

        cls_score = conv2d(x_cls, self.num_anchors * self.num_classes,
                                      kernel=3,
                                      bias = tf.constant_initializer(-math.log((1 - self.prior_prob) / self.prior_prob)),
                                      is_training=self.is_training,
                                      scope='ClsScoreConv')
        bbox_pred = conv2d(x_reg, self.num_anchors * 4,
                                      kernel=3,
                                      is_training=self.is_training,
                                      scope='BboxPredConv')
        object_pred = conv2d(x_reg, self.num_anchors,
                                      kernel=3,
                                      is_training=self.is_training,
                                      scope='ObjectPredConv')
        shape=tf.shape(cls_score)
        N, H, W = shape[0], shape[1], shape[2]
        cls_score = tf.reshape(cls_score, [N, -1, self.num_classes, H, W])
        objectness = tf.identity(object_pred)
        implicit objectness
        objectness = tf. reshape(objectness, [N, -1, 1, H, W])
        normalized_cls_score = cls_score + objectness - tf.log(
            1. + tf.clip_by_value(tf.exp(objectness), -self.INF, self.INF) + tf.clip_by_value(
                tf.exp(objectness), -self.INF, self.INF))
        normalized_cls_score = tf.reshape(normalized_cls_score, [N, -1, self.num_classes])
#         normalized_cls_score = tf.reshape(cls_score, [N, -1, self.num_classes])
        # print('normalized_cls_score', normalized_cls_score)
        
        return normalized_cls_score, bbox_pred, [[H,W]]

    def forward(self, x):
        normalized_cls_score, bbox_pred, feature_shape = self.build_layers(x)
        return normalized_cls_score, bbox_pred, feature_shape
        





        

