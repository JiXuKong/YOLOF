import json
import os
import sys
import tensorflow as tf
import cv2
import copy
import numpy as np
sys.path.append("/project/train/src_repo/FCOS")
from model import fcos
import config as cfg
from tensorflow.python.framework import graph_util

restore_path = '/project/train/models/model.ckpt-20000'
def build_graph(batch_size = 1, class_num = 7, image_size = 512):
    input_ = tf.placeholder(tf.float32, shape = [1, cfg.image_size_h, cfg.image_size_w, 3], name='Placeholder_0')
    out = fcos.FCOS(False).forward(input_)
    if cfg.cnt_branch:
        pred_class, pred_cnt, pred_reg = out
    else:
        pred_class, pred_reg = out
    _cls_scores, _cls_classes, _boxes = fcos.DetectHead(cfg.score_threshold, cfg.nms_iou_threshold, cfg.max_detection_boxes_num, cfg.strides).forward(out)
    boxes, scores, label = _boxes[0], _cls_scores[0], _cls_classes[0]#for one img

    boxes = tf.concat(boxes, axis = 0, name = 'boxes')
    scores = tf.concat(scores, axis = 0, name = 'scores')
    labels = tf.concat(label, axis = 0, name = 'labels')
    return boxes, scores, labels

pb_file_path = '/project/train/models'
with tf.Session(graph=tf.Graph()) as sess:
    boxes, scores, labels = build_graph()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, restore_path)
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['boxes', 'scores', 'labels'])
    with tf.gfile.FastGFile(pb_file_path+'/FCOS_res50_ckpt20000_Frozen.pb', mode='wb') as f:
        f.write(constant_graph.SerializeToString())
        print(pb_file_path+'/FCOS_res50_ckpt20000_Frozen.pb')

class TOD(object):
    def __init__(self, PATH_TO_CKPT):
        self.PATH_TO_CKPT = PATH_TO_CKPT
        self.detection_graph = self._load_model()
        self.image_tensor, self.boxes, self.scores, self.labels, self.sess = self.graph()

    def _load_model(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()#定义图
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')#导入图
        return detection_graph

    
    def graph(self):
        with self.detection_graph.as_default():
            sess = tf.Session(graph=self.detection_graph)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_tensor = self.detection_graph.get_tensor_by_name('Placeholder_0:0')
#             print(2)
            boxes = self.detection_graph.get_tensor_by_name('boxes:0')
#             print(2)
            scores = self.detection_graph.get_tensor_by_name('scores:0')
            labels = self.detection_graph.get_tensor_by_name('labels:0')
#             print(2)
            return image_tensor, boxes, scores, labels, sess    
    
    def detect(self, image):
        # Actual detection.
        boxes, scores, labels = self.sess.run(
            [self.boxes, self.scores, self.labels],
            feed_dict={self.image_tensor: image})    
        return boxes, scores, labels



def process(img):
    image_size_h = 512
    image_size_w = 640

    min_side, max_side    = image_size_h, image_size_w
    h, w, _  = img.shape
    scale = min_side/h
    if scale*w>max_side:
        scale = max_side/w
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(img, (nw, nh))
    paded_w = (image_size_w - nw)//2
    paded_h = (image_size_h - nh)//2

    image_paded = np.zeros(shape=[image_size_h, image_size_w, 3],dtype=np.uint8)
    image_paded[paded_h:(paded_h+nh), paded_w:(paded_w+nw), :] = image_resized
    img = image_paded
    
    img_orig = copy.deepcopy(img[:,:,::-1])
    img_orig = np.expand_dims(img_orig, axis=0)

    transform_delta = [-paded_w, -paded_h]
    transform_scale = [w/nw, h/nh]
    return img_orig, transform_delta, transform_scale



# Replace your own target label here.
label_id_map = {
    1: "slagcar",
    0: "non_slagcar"
}


def init():
    """Initialize model
    Returns: model
    """

    detecotr = TOD('/project/train/models/FCOS_res50_ckpt20000_Frozen.pb')
    return detecotr


def process_image(handle, input_image, thresh=0.3, args=None, ** kwargs):
    """Do inference to analysis input_image and get output

    Attributes:
        net: model handle
        input_image (numpy.ndarray): image to be process, format: (h, w, c), BGR
        thresh: thresh value

    Returns: process result

    """

    # ------------------------------- Prepare input -------------------------------------
    input_image, transform_delta, transform_scale = process(input_image)

    # --------------------------- Performing inference ----------------------------------
    boxes, scores, labels = handle.detect(input_image)

    # --------------------------- Read and postprocess output ---------------------------
    result = {}
    object = []
    for j in range(boxes.shape[0]):
        if scores[j] >=thresh :
            if labels[j] == 1:
                l = 1
            else:
                l = 0
            b = boxes[j]
            x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
            x1 = (x1 + transform_delta[0])*transform_scale[0]
            x2 = (x2 + transform_delta[0])*transform_scale[0]
            y1 = (y1 + transform_delta[1])*transform_scale[1]
            y2 = (y2 + transform_delta[1])*transform_scale[1]
            category = label_id_map[l]
            width, height = x2-x1, y2-y1
            cent_x, cent_y = (x1 + x2)/2, (y1 + y2)/2
            object.append({
                "x": int(x1),
                "y": int(y1),
                "height": int(y2-y1),
                "width": int(x2-x1),
                "confidence": float(scores[j]),
                "name": category
            })
    
    result["algorithm_data"] = {
        "is_alert": False,
        "target_count": 0,
        "target_info": []
    }
    # result["objects"] = object
    result["model_data"]={"objects": object}
    # print(result)
    return json.dumps(result, indent = 4)
    
            

    # classes_prob = np.squeeze(classes_prob)
    # detect_result = {'class': label_id_map[np.argmax(classes_prob)]}
    # return json.dumps(detect_result)


# print('done')
# if __name__ == '__main__':
#     """Test python api
#     """
#     img = cv2.imread('/home/data/14/punk/4UqguWG3IQ475shj42JH.jpg')
#     predictor = init()
#     result = process_image(predictor, img, 0.5)
#     log.info(result)
