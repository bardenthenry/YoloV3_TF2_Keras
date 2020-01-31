import numpy as np
from utils.DataProcess import RandomHSV, RandomBlur, RandomResize, RandomFlip, RandomRotate, ResizeOrCropToInputSize, BoxToTensor
import os
import random
import tensorflow as tf

class ImageData():
    def __init__(self, input_shape, class_ls, anchor_ls, anchor_mask, reduce_ratio,
                 hsv_delta, q_delta, resize_scale_range, flip_mode, angle_range, resize_method = "lanczos3", random = True, test_acc_mode = False):
        self.random        = random
        self.test_acc_mode = test_acc_mode
        
        self.random_hsv           = RandomHSV(hsv_delta)
        self.random_blur          = RandomBlur(q_delta)
        self.random_resize        = RandomResize(resize_scale_range, resize_method)
        self.random_flip          = RandomFlip(flip_mode)
        self.random_rotate        = RandomRotate(angle_range)
        self.img_box_to_inputsize = ResizeOrCropToInputSize(input_shape, resize_method, random)
        self.box_to_tensor        = BoxToTensor(input_shape, class_ls, anchor_ls, anchor_mask, reduce_ratio)
        
    def TF_DataPreprocess(self, img, boxes):
        if self.random:
            img = self.random_hsv(img)
            img = self.random_blur(img)
            img, boxes = self.random_resize(img, boxes)
            img, boxes = self.random_flip(img, boxes)
            img, boxes = self.random_rotate(img, boxes)
            
        img, boxes = self.img_box_to_inputsize(img, boxes)
        
        img = tf.dtypes.cast(img, tf.float32)
        # img = tf.clip_by_value(img, 0., 255.)
        
        if self.test_acc_mode:
            return img / 255., boxes
            
        else:
            y_true_0, y_true_1, y_true_2 = self.box_to_tensor(boxes)

            return img / 255., (y_true_0, y_true_1, y_true_2) #boxes[:1,...]
        
    def TF_Parser(self, record):
        '''
        TFRecordDataset 的解析器
        '''
        img_features = tf.io.parse_single_example(
            record,
            features = {
                'height'      : tf.io.FixedLenFeature([], tf.int64),
                'width'       : tf.io.FixedLenFeature([], tf.int64),
                'depth'       : tf.io.FixedLenFeature([], tf.int64),
                'image_raw'   : tf.io.FixedLenFeature([], tf.string),
                'boxes_height': tf.io.FixedLenFeature([], tf.int64),
                'boxes_weight': tf.io.FixedLenFeature([], tf.int64),
                'boxes'       : tf.io.VarLenFeature(tf.float32)
            }
        )
        is_jpg = tf.io.is_jpeg(img_features['image_raw'])
        image = tf.cond(
            is_jpg,
            lambda: tf.io.decode_jpeg(img_features['image_raw']),
            lambda: tf.io.decode_png(img_features['image_raw'])
        )
        boxes = tf.sparse.to_dense(img_features['boxes'])
        boxes = tf.reshape(boxes, [img_features['boxes_height'], img_features['boxes_weight']])

        return image, boxes
    
    def CreateDataset(self, tfrecord_file, batch_size, epochs = 1, shuffle_size = None, train = True, num_parallel_reads = None, num_parallel_calls = None):
        # 讀取 TFRecord
        self.dataset = tf.data.TFRecordDataset(tfrecord_file, num_parallel_reads)

        # 解析 TFRecord
        self.dataset = self.dataset.map(self.TF_Parser) #.cache()
        
        # 資料前處理流程
        self.dataset = self.dataset.map(self.TF_DataPreprocess, num_parallel_calls = num_parallel_calls)
        
        # 定義 epochs shuffle_size batch_size
        if train:    
            self.dataset = self.dataset.shuffle(buffer_size=shuffle_size)
        
        self.dataset = self.dataset.batch(batch_size)
        #self.dataset = self.dataset.prefetch(buffer_size = batch_size * 1)
        
        if epochs > 1:
            self.dataset = self.dataset.repeat(epochs)
        
        