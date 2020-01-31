from utils.ImgDataInfo import ImgDataInfo
import json
import os
import random

def main():
    # read config.json
    with open('config.json') as json_file:
        cfg = json.load(json_file)
        
    data_dir = './data'
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    
    img_ls = os.listdir(cfg['img_dir'])
    
    # read xml to img info obj
    img_info_ls = []
    for img in img_ls:
        try:
            img_info = ImgDataInfo(img, cfg['img_dir'], cfg['annotation_dir'])(cfg["class_ls"])
            img_info_ls.append(img_info)
        except:
            None
    
    # shuffle
    random.shuffle(img_info_ls)
    
    # split training and validation
    n_data = len(img_info_ls)
    val_num = int(n_data * cfg["val_train_ratio"])
    
    val_info_ls = img_info_ls[:val_num]
    train_info_ls = img_info_ls[val_num:]
    
    with open(os.path.join(data_dir, cfg["train_img_info"]), 'w') as outfile:
        json.dump(train_info_ls, outfile)
        
    with open(os.path.join(data_dir, cfg["val_img_info"]), 'w') as outfile:
        json.dump(val_info_ls, outfile)
        
if __name__=="__main__":
    main()
    