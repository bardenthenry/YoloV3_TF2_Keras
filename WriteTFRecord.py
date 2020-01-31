from utils.ImgDataInfo import ImgDataInfo
import tensorflow as tf
import numpy as np
import cv2
import random
import json
import os
from tqdm import trange

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def img_boxes_example(img_info):
    image_string = open(img_info["img_path"], 'rb').read()
    image_shape = tf.image.decode_image(image_string).shape
    
    boxes = np.array(img_info["boxes"], np.float32)
    boxes_shape = list(boxes.shape)
    boxes_height = boxes_shape[0]
    boxes_weight = boxes_shape[1]
    boxes_flatten = boxes.reshape(-1).tolist()
    
    feature = {
        'height'      : _int64_feature(image_shape[0]),
        'width'       : _int64_feature(image_shape[1]),
        'depth'       : _int64_feature(image_shape[2]),
        'image_raw'   : _bytes_feature(image_string),
        'boxes_height': _int64_feature(boxes_height),
        'boxes_weight': _int64_feature(boxes_weight),
        'boxes'       : tf.train.Feature(float_list=tf.train.FloatList(value=boxes_flatten))
    }
    
    return tf.train.Example(features=tf.train.Features(feature=feature))

def write_to_tfrecord(img_info_ls, record_file):
    with tf.io.TFRecordWriter(record_file) as writer:
        t = trange(len(img_info_ls), ncols = 100)
        for i in t:
            img_info = img_info_ls[i]
            example = img_boxes_example(img_info)
            writer.write(example.SerializeToString())
    
def main():
    data_dir = './data'
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    
    conf_dir = './config'
    
    print("Reading Model Config File...")
    model_cfg_json_file = os.path.join(conf_dir, 'model_config.json')
    with open(model_cfg_json_file) as j:
        model_cfg = json.load(j)
    
    print("Reading Dataset Config File...")
    dataset_cfg_json_file = os.path.join(conf_dir, 'dataset_config.json')
    with open(dataset_cfg_json_file) as j:
        dataset_cfg = json.load(j)
        
    img_ls = os.listdir(dataset_cfg['img_dir'])
    
    class_ls = model_cfg["class_ls"]
    
    print("Classes:", end = " ")
    print(class_ls)
    # read xml to img info obj
    img_info_ls = []
    
    for img in img_ls:
        try:
            img_info = ImgDataInfo(img, dataset_cfg['img_dir'], dataset_cfg['annotation_dir'])(class_ls)
            
            if len(img_info["boxes"]) > 0:
                img_info_ls.append(img_info)
                
        except:
            None

    # shuffle
    random.shuffle(img_info_ls)
    
    # split training and validation
    n_data = len(img_info_ls)
    val_num = int(n_data * dataset_cfg["val_train_ratio"])

    valid_info_ls = img_info_ls[:val_num]
    train_info_ls = img_info_ls[val_num:]
    
    train_records = os.path.join(data_dir, "train.tfrecords")
    valid_records = os.path.join(data_dir, "val.tfrecords")
    write_to_tfrecord(train_info_ls, train_records) # write train data
    write_to_tfrecord(valid_info_ls, valid_records) # write valid data
    
    num_data_dic = {
        "train": len(train_info_ls),
        "val"  : len(valid_info_ls)
    }
    
    with open(os.path.join(data_dir, 'num_data.json'), 'w') as outfile:
        json.dump(num_data_dic, outfile)
    
if __name__ == "__main__":
    main()
    