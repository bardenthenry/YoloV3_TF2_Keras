import tensorflow as tf
from net.Module import DarknetModule, DarknetModule, YoloModule, ScaleOutputBox, ComputeIgnoreMask, YoloLoss, NMS

from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, LeakyReLU, Add, UpSampling2D, Reshape
from tensorflow.keras.layers import concatenate as Concate
from tensorflow.keras.regularizers import l2

    
class YoloV3():
    def __init__(self, input_shape, class_ls, anchor_ls, anchor_mask, reduce_ratio, iou_thresh = 0.5, l2_decay = 5e-4, alpha = 0.1, class_method = 'sigmoid'):
        predict_shape = [[input_shape[0] // i, input_shape[1] // i, len(anchor_ls) // 3, 5 + len(class_ls)] for i in reduce_ratio]
        num_anchor  = len(anchor_mask[0])
        filters_out = num_anchor * (5 + len(class_ls))
        self.iou_thresh = iou_thresh
        self.input_shape = input_shape
        self.anchor_tensor_ls = [tf.constant([anchor_ls[j] for j in i]) for i in anchor_mask]
        self.class_method = class_method
        
        # x_shape_wh y_shape_wh
        self.x_shape_wh = tf.constant(input_shape[::-1])
        self.y_shape_wh_ls = [tf.constant(shape[:2][::-1]) for shape in predict_shape]
        
        # start conv
        self.first_conv = Conv2D(32, 3, (1,1), 'same', use_bias = False, kernel_regularizer = l2(l=l2_decay))
        self.first_bn   = BatchNormalization(trainable=True)
        self.first_act  = LeakyReLU(alpha)
        
        # darknet53
        self.dark_block_0 = DarknetModule(32,  alpha, l2_decay, 1)
        self.dark_block_1 = DarknetModule(64,  alpha, l2_decay, 2)
        self.dark_block_2 = DarknetModule(128, alpha, l2_decay, 8)
        self.dark_block_3 = DarknetModule(256, alpha, l2_decay, 8)
        self.dark_block_4 = DarknetModule(512, alpha, l2_decay, 4)
        
        # yolo
        self.yolo_block_0 = YoloModule(512, filters_out, alpha, l2_decay)
        self.yolo_block_1 = YoloModule(256, filters_out, alpha, l2_decay)
        self.yolo_block_2 = YoloModule(128, filters_out, alpha, l2_decay, end = True)
        
        self.reshape_y_0 = Reshape((predict_shape[0][0], predict_shape[0][1], num_anchor, 5 + len(class_ls)), name = 'y0')
        self.reshape_y_1 = Reshape((predict_shape[1][0], predict_shape[1][1], num_anchor, 5 + len(class_ls)), name = 'y1')
        self.reshape_y_2 = Reshape((predict_shape[2][0], predict_shape[2][1], num_anchor, 5 + len(class_ls)), name = 'y2')
        
    def build_network(self, input_tensor):
        x = self.first_conv(input_tensor)
        x = self.first_bn(x)
        x = self.first_act(x)
        
        x = self.dark_block_0(x)
        x = self.dark_block_1(x)
        dark2 = self.dark_block_2(x)
        dark3 = self.dark_block_3(dark2)
        dark4 = self.dark_block_4(dark3)
        
        x, y0 = self.yolo_block_0(dark4)
        x     = Concate([x, dark3], axis = -1, name = 'concate_dark3')
        
        x, y1 = self.yolo_block_1(x)
        x     = Concate([x, dark2], axis = -1, name = 'concate_dark2')
        
        x, y2 = self.yolo_block_2(x)
        
        y0 = self.reshape_y_0(y0)
        y1 = self.reshape_y_1(y1)
        y2 = self.reshape_y_2(y2)
        
        return y0, y1, y2
        
    def build_model(self, learning_rate, clipping):
        input_tensor = Input(shape = (self.input_shape[0], self.input_shape[1], 3))
        y0, y1, y2 = self.build_network(input_tensor)
        
        loss_fun = {
            'y0': YoloLoss(self.anchor_tensor_ls[0], self.x_shape_wh, self.iou_thresh, self.class_method),
            'y1': YoloLoss(self.anchor_tensor_ls[1], self.x_shape_wh, self.iou_thresh, self.class_method),
            'y2': YoloLoss(self.anchor_tensor_ls[2], self.x_shape_wh, self.iou_thresh, self.class_method)
        }
        if clipping is None:
            optimizer = Adam(learning_rate, amsgrad = True)
        else:
            optimizer = Adam(learning_rate, clipnorm = clipping, amsgrad = True)
            
        model = Model(inputs=input_tensor, outputs= [y0, y1, y2], name='yolov3')
        model.compile(optimizer, loss = loss_fun)
        model.summary()
        
        return model
        
    def load_model(self, save_path):
        input_tensor = Input(shape = (self.input_shape[0], self.input_shape[1], 3))
        orign_img_hw = Input(shape=(2), dtype = tf.float32, name = 'origin_image_shape') # 新增 original img shape 的 input, dim => (1,2) 
        iou_thresh   = Input(shape=(1), dtype = tf.float32, name = 'iou_thresh')         # 新增 iou_thresh 的 input, dim => (1,1)
        score_thresh = Input(shape=(1), dtype = tf.float32, name = 'score_thresh')       # 新增 score_thresh 的 input, dim => (1,1)
        
        y_ls = self.build_network(input_tensor)
        
        # 將 output 轉換成正確的尺度
        box_ls, score_ls = [], []
        for i in range(len(y_ls)):
            p_box, p_score = ScaleOutputBox(self.anchor_tensor_ls[i], tf.constant(self.input_shape[::-1], dtype = tf.float32), self.class_method, False)(y_ls[i], orign_img_hw)

            box_ls.append(p_box)
            score_ls.append(p_score)
            
        boxes, scores = tf.concat(box_ls, 1), tf.concat(score_ls, 1)
                
        # NMS
        nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = NMS(iou_thresh, score_thresh, self.input_shape)(boxes, scores, orign_img_hw) #nms_result = (nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections)
        ymin, xmin, ymax, xmax = tf.split(nmsed_boxes, [1,1,1,1], axis = -1)
        correct_box = tf.concat([xmin, ymin, xmax, ymax], -1)
        
        predict_model = tf.keras.Model(
            inputs = [input_tensor, orign_img_hw, iou_thresh, score_thresh], 
            outputs = [correct_box, nmsed_scores, nmsed_classes, valid_detections], 
            name='PredictModel'
        )
        
        predict_model.load_weights(save_path)
        
        return predict_model
