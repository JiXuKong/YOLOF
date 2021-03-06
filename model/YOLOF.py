import tensorflow as tf
import numpy as np
import math
from model.backbone.shufflenet.shufflenet_v2 import shufflenet_v2
from model.backbone.resnet.resnet_ import resnet_base
from model.tool.anchor import generate_anchor_
from model.tool.uniform_matcher import OneImgUniformMatcher
from model.tool.regress_target import reverse_regress_target_tf#, normolize_box, reverse_normolize_box
from model.tool.NMS import gpu_nms
from model.encoder import DilatedEncoder
from model.decoder import Decoder
import config as cfg

slim=tf.contrib.slim
class YOLOF(object):
    def __init__(self, base_anchor, scale, aspect_ratio, class_num, is_training):
        self.base_anchor=base_anchor
        self.scale=scale
        self.aspect_ratio=aspect_ratio
        self.is_training=is_training
        self.class_num=class_num


    def _generate_anchor(self, feature_shape, base_anchor, scale, aspect_ratio):
        anchorlist = []
        for i in range(len(base_anchor)):
            anchors = generate_anchor_(base_anchor[i], scale, aspect_ratio, feature_shape[i])
            anchorlist.append(anchors)
        anchor = np.concatenate(anchorlist, axis = 0).astype(np.float32)
        # print("anchor", anchor)
        return anchor
        
    def forward(self, input_):
        # end_points = resnet_base(input_, self.is_training, 'resnet_v1_50')
        end_points = resnet_base(input_, self.is_training, 'resnet_v1_50')
        with tf.variable_scope("DilatedEncoder"):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(0.0001)):
                fpn_features = DilatedEncoder(encoder_channels=cfg.encoder_channels,
                    block_mid_channels=cfg.block_mid_channels,
                    num_residual_blocks=cfg.num_residual_blocks,
                    block_dilations=cfg.block_dilations,
                    is_training=self.is_training).build_layers(end_points['p5'])
        with tf.variable_scope("Decoder"):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(0.0001)):
                normalized_cls_score, bbox_pred, feature_shape = Decoder(in_channels=cfg.encoder_channels,
                    num_classes=self.class_num,
                    num_anchors=cfg.anchors,
                    cls_num_convs=cfg.cls_num_convs,
                    reg_num_convs=cfg.reg_num_convs,
                    prior_prob=cfg.pi,
                    is_training=self.is_training).forward(fpn_features)
            

        all_anchors = tf.py_func(self._generate_anchor,
            inp=[feature_shape, self.base_anchor, self.scale, self.aspect_ratio],
            Tout=[tf.float32]
            )
        # print("all_anchors", all_anchors)
        return [normalized_cls_score, bbox_pred], all_anchors

class DetectHead(object):
    def __init__(self,score_threshold,nms_iou_threshold,max_detection_boxes_num,strides=None):
        self.score_threshold=score_threshold
        self.nms_iou_threshold=nms_iou_threshold
        self.max_detection_boxes_num=max_detection_boxes_num
        self.strides=strides
    def forward(self,inputs_0, inputs_1, anchor, imgsize, class_num, score_thresh, nms_thresh):
        '''

        inputs  list [normalized_cls_score, bbox_pred]  
        normalized_cls_score  [batch_size, h*w, class_num]  
        bbox_pred  [batch_size, h*w, anchor_num*4]  
        '''
        # print('print', print)
        normalized_cls_score, bbox_pred = tf.stop_gradient(inputs_0), tf.stop_gradient(inputs_1)
        normalized_cls_score = tf.sigmoid(normalized_cls_score)
        _cls_scores=[]
        _cls_classes=[]
        _boxes=[]
        for batch in range(bbox_pred.get_shape().as_list()[0]):
            single_predbox = reverse_regress_target_tf(tf.reshape(bbox_pred[batch], [1, -1, 4]), anchor, [imgsize, imgsize])
            single_score = tf.reshape(normalized_cls_score[batch], [1, -1, class_num])
            nms_box, nms_score, nms_label = gpu_nms(single_predbox, single_score, 
                class_num, max_boxes=50, score_thresh=score_thresh, nms_thresh=nms_thresh)
            _cls_scores.append(nms_score)
            _cls_classes.append(nms_label)
            _boxes.append(nms_box)
        # _cls_scores = tf.stack(_cls_scores, axis = 0)#[batch, max_num]
        # _cls_classes = tf.stack(_cls_classes, axis = 0)#[batch, max_num]
        # _boxes = tf.stack(_boxes, axis = 0)#[batch, max_num, 4]
        return _cls_scores, _cls_classes, _boxes
        
