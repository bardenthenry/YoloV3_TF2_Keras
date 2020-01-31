import os
import tensorflow as tf
import json
import numpy as np

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TerminateOnNaN
from utils.ReadDataFromTFRecord import ImageData
from net.Network import YoloV3
from net.Module import GradientCallback

def main():
    data_dir = './data'
    conf_dir = './config'
    
    print("Reading Model Config File...")
    model_cfg_json_file = os.path.join(conf_dir, 'model_config.json')
    with open(model_cfg_json_file) as j:
        model_cfg = json.load(j)
    
    print("Reading Dataset Config File...")
    dataset_cfg_json_file = os.path.join(conf_dir, 'dataset_config.json')
    with open(dataset_cfg_json_file) as j:
        dataset_cfg = json.load(j)
        
    print("Reading Train Config File...")
    train_cfg_json_file = os.path.join(conf_dir, 'train_config.json')
    with open(train_cfg_json_file) as j:
        train_cfg = json.load(j)
        
    # Set Enviroment
    if train_cfg["CUDA_VISIBLE_DEVICES"] is not None:
        print("Set CUDA_VISIBLE_DEVICES = {}".format(train_cfg["CUDA_VISIBLE_DEVICES"]))
        os.environ["CUDA_VISIBLE_DEVICES"] = train_cfg["CUDA_VISIBLE_DEVICES"]
        
    if not train_cfg["parallel_mode"]:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    # Make Log Dir
    log_dir = train_cfg["log_dir"]
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    
    
    if not os.path.isdir(train_cfg["model_save_path"]):
        os.makedirs(train_cfg["model_save_path"])
        
    
    # Read Number of Train and Valid Data
    print("Reading Number of Train and Valid Data...")
    num_data_file = os.path.join(data_dir, "num_data.json") 
    with open(num_data_file) as j:
        num_data = json.load(j)

    num_train = num_data["train"]
    num_val = num_data["val"]

    # Set Train and Validation Dataset
    print("Setting Train and Validation Dataset...")
    train_data = ImageData(
        model_cfg['input_shape'],
        model_cfg['class_ls'],
        model_cfg['anchor_ls'],
        model_cfg['anchor_mask'],
        model_cfg['reduce_ratio'],
        dataset_cfg['hsv_delta'],
        dataset_cfg['q_delta'],
        dataset_cfg['resize_scale_range'],
        dataset_cfg['flip_mode'],
        dataset_cfg['angle_range'],
        dataset_cfg['resize_method'],
        True
    )

    val_data = ImageData(
        model_cfg['input_shape'],
        model_cfg['class_ls'],
        model_cfg['anchor_ls'],
        model_cfg['anchor_mask'],
        model_cfg['reduce_ratio'],
        dataset_cfg['hsv_delta'],
        dataset_cfg['q_delta'],
        dataset_cfg['resize_scale_range'],
        dataset_cfg['flip_mode'],
        dataset_cfg['angle_range'],
        dataset_cfg['resize_method'],
        False
    )

    train_data.CreateDataset(
        os.path.join(data_dir, 'train.tfrecords'), 
        batch_size = train_cfg['batch_size'], 
        epochs = train_cfg["n_epoc"], 
        shuffle_size = 10, 
        train = True, 
        num_parallel_calls = train_cfg['num_parallel_calls']
    )
    
    val_data.CreateDataset(
        os.path.join(data_dir, 'val.tfrecords'), 
        batch_size = train_cfg['batch_size'], 
        epochs = train_cfg["n_epoc"], 
        train = False, 
        num_parallel_calls = train_cfg['num_parallel_calls']
    )
    
    # Set Training Stratege
    print("Training Start...")
    logging = TensorBoard(log_dir = log_dir, update_freq = 'batch')
    
    checkpoint = ModelCheckpoint(
        os.path.join(log_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'), 
        monitor='val_loss', 
        save_weights_only=True, 
        save_best_only=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.1, 
        patience=train_cfg["lr_patient"], 
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        min_delta=0, 
        patience=train_cfg["early_stop_patient"], 
        verbose=1
    )
    
    nan_terminate = TerminateOnNaN()
    
#     file_writer = tf.summary.create_file_writer("./metrics")
#     file_writer.set_as_default()
#     gradient_cb = GradientCallback()
    
    step_train = num_train // train_cfg['batch_size']
    step_val = num_val // train_cfg['batch_size']

    # Build YoloV3 Model
    print("Building YoloV3 Model...")
    yolo = YoloV3(
        model_cfg['input_shape'],
        model_cfg['class_ls'],
        model_cfg['anchor_ls'],
        model_cfg['anchor_mask'],
        model_cfg['reduce_ratio'],
        model_cfg['iou_thresh'],
        model_cfg['l2_decay'],
        model_cfg['alpha'],
        model_cfg['class_method']
    )
    # set mixed percision
    # tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
    
    if not train_cfg["parallel_mode"]:
        model = yolo.build_model(train_cfg['learning_rate'], train_cfg["clipvalue"])
        
        # Training Start
        model.fit(
            train_data.dataset, epochs = train_cfg["n_epoc"], 
            steps_per_epoch = step_train,
            validation_data = val_data.dataset,
            validation_steps = step_val,
            callbacks = [logging, checkpoint, reduce_lr, early_stopping, nan_terminate]
        )
        
    else:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = yolo.build_model(train_cfg['learning_rate'], train_cfg["clipvalue"])
            
            # Training Start
            model.fit(
                train_data.dataset, epochs = train_cfg["n_epoc"], 
                steps_per_epoch = step_train,
                validation_data = val_data.dataset,
                validation_steps = step_val,
                callbacks = [logging, checkpoint, reduce_lr, early_stopping]
            )
            
    model.save_weights(os.path.join(train_cfg["model_save_path"])+ "/Yolo.h5")
                       
if __name__ == "__main__":
    main()