import xml.etree.ElementTree as ET
import os

class ImgDataInfo():
    def __init__(self, img_name, img_dir, xml_dir):
        self.img_name    = img_name
        self.img_dir     = img_dir
        self.file_name   = img_name.split('.')[0]
        self.img_format  = img_name.split('.')[-1]
        self.boxes       = []
        self.xml_dir    = xml_dir
    
    def __call__(self, class_ls):
        '''
        read xml file and turn it to np.array
        Input Parameter:
        file: (string) xml path

        Output Object
        np.array([
            [xmin, ymin, xmax, ymax, confidence, classes],
            [xmin, ymin, xmax, ymax, confidence, classes],
            ...
        ])
        '''
        file = os.path.join(self.xml_dir, '{}.{}'.format(self.file_name, 'xml'))
        # xml to obj array
        tree = ET.ElementTree(file = file)
        root = tree.getroot()

        # Find object key location and get the min and max of x and y
        for obj in root.findall('object'):
            confidence = 1.
            class_name = obj.find('name').text

            if class_name in class_ls: # check if class in list or not
                class_index = class_ls.index(class_name)
                for bndbox in obj.findall('bndbox'):
                    xmin = int(float(bndbox.find('xmin').text))
                    ymin = int(float(bndbox.find('ymin').text))
                    xmax = int(float(bndbox.find('xmax').text))
                    ymax = int(float(bndbox.find('ymax').text))
                
                obj_array = [xmin, ymin, xmax, ymax, confidence, class_index]
                self.boxes.append(obj_array)
            
            else:
                next
        
        out_dic = {
            "img_path"   : os.path.join(self.img_dir, self.img_name),
            "img_format" : self.img_format,
            "boxes"      : self.boxes
        }
        
        return out_dic