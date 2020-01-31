import xml.etree.ElementTree as ET
from os import getcwd
import os
import json

# Read Config
conf_dir = './config'
    
print("Reading Model Config File...")
model_cfg_json_file = os.path.join(conf_dir, 'model_config.json')
with open(model_cfg_json_file) as j:
    model_cfg = json.load(j)
    
print("Reading Dataset Config File...")
dataset_cfg_json_file = os.path.join(conf_dir, 'dataset_config.json')
with open(dataset_cfg_json_file) as j:
    dataset_cfg = json.load(j)
    
classes = model_cfg["class_ls"]

annotation_dir = dataset_cfg["annotation_dir"]
image_dir = dataset_cfg["img_dir"]

def convert_annotation(image_id, list_file):
    xml_path = os.path.join(annotation_dir, '%s.xml')%(image_id)
    in_file = open(xml_path)
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()

image_file_ls = [file for file in os.listdir(image_dir) if 'jpg' in file]
image_ids_ls = [file.split('.')[0] for file in image_file_ls]
list_file = open('img_path_bonding_box.txt', 'w')

for image_id in image_ids_ls:
    img_path = os.path.join(image_dir, "{}.jpg".format(image_id))
    list_file.write(img_path)
    convert_annotation(image_id, list_file)
    list_file.write('\n')
list_file.close()

