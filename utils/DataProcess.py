import numpy as np
import random
import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image, ImageDraw 

def ReadImgFile(img_path):
    '''
    read image
    Input Parameter:
    img_path: <string> img path
    '''
    # read image
    image_raw = tf.io.read_file(img_path)
    img = tf.image.decode_image(image_raw)

    return img

def DropInvalidBox(img, boxes):
    i_wh = tf.dtypes.cast(tf.shape(img)[::-1][1:3], tf.float32)
    # 校正超出 n_w, n_h 的座標
    xmin, ymin, xmax, ymax = tf.split(boxes[...,:4], [1,1,1,1], -1)
    xmin = tf.clip_by_value(xmin, 0., i_wh[0]-1.)
    ymin = tf.clip_by_value(ymin, 0., i_wh[1]-1.)
    xmax = tf.clip_by_value(xmax, 0., i_wh[0]-1.)
    ymax = tf.clip_by_value(ymax, 0., i_wh[1]-1.)
    
    new_boxes = tf.concat([xmin, ymin, xmax, ymax, boxes[..., 4:]], axis = -1) #[x,y,w,h,c...]
    boxes_wh = new_boxes[..., 2:4] - new_boxes[..., :2]
    boxes_area = boxes_wh[:, 0] * boxes_wh[:, 1]
    # w h < 5 的物件
    mask = tf.math.logical_and(boxes_wh[:, 0] > 5., boxes_wh[:,1] > 5.)
    new_boxes = tf.boolean_mask(new_boxes, mask)
    
    return new_boxes

def FramedObject(img, boxes, class_ls):
    '''
    Framed the obj in the img
    '''
    labeled_img = Image.fromarray(img)
    draw_img = ImageDraw.Draw(labeled_img)  
    for i in range(boxes.shape[0]):
        # object box info
        obj_box   = boxes[i, :4].tolist()
        obj_score = boxes[i, 4]
        obj_class_idx = boxes[i, 5].astype(np.int32)
        obj_class = class_ls[obj_class_idx]
        
        # object class info
        score_per = np.round(obj_score * 100, 2)
        text      = '{} {}%'.format(obj_class, str(score_per))

        text_len = len(text)

        text_rect_min = tuple((boxes[i, :2] + [0, -10]).tolist())
        text_rect_max = tuple((boxes[i, :2] + [text_len * 6.5, 0]) .tolist())
        text_min = tuple((boxes[i, :2] + [2, -10]).tolist())
        
        # label box and write class info
        draw_img.rectangle(obj_box, fill = None, outline =(0,255,0,128))
        draw_img.rectangle([text_rect_min, text_rect_max], fill = (0,255,0,128), outline = (0,255,0,128))
        draw_img.text(text_min, text, (0,0,0,128))
        
    return labeled_img

def BoxIOU(box_a, box_b):
    '''
    box_a, box_b: np.array with shape (4,)
    '''
    a_w = box_a[2] - box_a[0]
    a_h = box_a[3] - box_a[1]
    a_area = a_w * a_h
    
    b_w = box_b[2] - box_b[0]
    b_h = box_b[3] - box_b[1]
    b_area = b_w * b_h
    
    i_min = np.maximum(box_a[:2],  box_b[:2])
    i_max = np.minimum(box_a[2:4], box_b[2:4])
    i_wh = i_max - i_min
    i_area = i_wh[ 0] * i_wh[ 1]
    
    u_area = a_area + b_area - i_area
    iou = i_area / u_area
    
    return iou

def SecondNMS(boxes, second_iou_thresh = 0.5):
    boxes = boxes[boxes[:,4].argsort()[::-1]]
    for i in range(boxes.shape[0]):
        if i >= boxes.shape[0]:
            break

        stay_ls = list(range(i + 1))
        for j in range(i+1, boxes.shape[0]):
            box_a = boxes[i,:]
            box_b = boxes[j,:]
            iou = BoxIOU(box_a, box_b)
            if iou < second_iou_thresh:
                stay_ls.append(j)

        boxes = boxes[stay_ls,:]
    
    return boxes

class RandomHSV():
    def __init__(self, hsv_delta):
        '''
        改變影像的 色相 飽和度 明度
        HSV 定義: 
        H: 0 ~ 179 色相
        S: 0 ~ 255 飽和度
        V: 0 ~ 255 明度

        hsv_delta: list(3), 0~1
        '''
        self.hsv_delta = hsv_delta
        
    def __call__(self, img):
        # img = tf.image.random_contrast(img, 0.1, 0.6)
        if self.hsv_delta[0] > 0:
            img = tf.image.random_hue(img, self.hsv_delta[0])
        
        if self.hsv_delta[1] > 0:
            img = tf.image.random_saturation(img, 1-self.hsv_delta[1], 0.9)
            
        if self.hsv_delta[2] > 0:
            img = tf.image.random_brightness(img, self.hsv_delta[2])
        
        return img
    
class RandomBlur():
    def __init__(self, q_delta):
        '''
        d_quality: 0 ~ 1
        '''
        self.q_delta = q_delta
        self.lower = int(100 * ( 1 - self.q_delta))
        
    def __call__(self, img):
        if self.q_delta > 0:
            im_shape = img.shape
            img.set_shape(im_shape)
            img = tf.image.random_jpeg_quality(img, self.lower, 100)
            
        return img
    
class RandomResize():
    def __init__(self, resize_scale_range, method):
        self.resize_scale_range = resize_scale_range
        self.method = method
        
    def ResizeBox(self, box, hw_scale, r_img):
        xmin, ymin, xmax, ymax = tf.split(box[..., :4], [1,1,1,1], -1)
        r_xmin = xmin * hw_scale[1]
        r_ymin = ymin * hw_scale[0]
        r_xmax = xmax * hw_scale[1]
        r_ymax = ymax * hw_scale[0]
        r_box = tf.concat([r_xmin, r_ymin, r_xmax, r_ymax, box[..., 4:]], -1)
        r_box = DropInvalidBox(r_img, r_box)
        
        return r_box
    
    def __call__(self, img, boxes):
        hw_scale = tf.random.uniform((2, ), self.resize_scale_range[0], self.resize_scale_range[1])
        n_hw = tf.dtypes.cast(hw_scale * tf.dtypes.cast(tf.shape(img)[:2], tf.float32), tf.int32)
        resize_img = tf.image.resize(img, n_hw, self.method)
        resize_box = tf.cond(tf.shape(boxes)[0] > 0, lambda : self.ResizeBox(boxes, hw_scale, resize_img), lambda : boxes)
        
        return resize_img, resize_box

class RandomFlip():
    def __init__(self, flip_mode):
        '''
        flip_mode: list with 1 or 2 (example: [1, 2]) 1 is left or right, 2 is up and down
        '''
        self.flip_mode = flip_mode
        
    def box_flip_left_right(self, box, w):
        xmin, ymin, xmax, ymax = tf.split(box[..., :4], [1,1,1,1], -1)
        xmin_f = w - xmax
        xmax_f = xmin_f + (xmax - xmin)
        
        box_f = tf.concat([xmin_f, ymin, xmax_f, ymax, box[..., 4:]], -1)
        
        return box_f
    
    def box_flip_up_down(self, box, h):
        xmin, ymin, xmax, ymax = tf.split(box[..., :4], [1,1,1,1], -1)
        ymin_f = h - ymax
        ymax_f = ymin_f + (ymax - ymin)
        
        box_f = tf.concat([xmin, ymin_f, xmax, ymax_f, box[..., 4:]], -1)
        
        return box_f
    
    def __call__(self, img, boxes):
        logits = tf.constant([[10., 10.]])
        num_samples = 2
        is_flip = tf.random.categorical(logits, num_samples, tf.int32)
        h, w, c = tf.split(tf.dtypes.cast(tf.shape(img), boxes.dtype), [1,1,1], 0)
        
        if 1 in self.flip_mode:
            flip_img = tf.cond(is_flip[0,0]==1, lambda : tf.image.flip_left_right(img), lambda : img)
            flip_boxes = tf.cond(tf.logical_and(is_flip[0,0]==1, tf.shape(boxes)[0] > 0), lambda: self.box_flip_left_right(boxes, w), lambda : boxes)
        
        if 2 in self.flip_mode:
            flip_img = tf.cond(is_flip[0,1]==1, lambda : tf.image.flip_up_down(img), lambda : img)
            flip_boxes = tf.cond(tf.logical_and(is_flip[0,1]==1, tf.shape(boxes)[0] > 0), lambda: self.box_flip_up_down(boxes, h), lambda : boxes)

        return flip_img, flip_boxes

class RandomRotate():
    def __init__(self, angle_range):
        '''
        angle_range: -10 * np.pi / 180 ~ 10 * np.pi / 180
        '''
        self.angle_range = angle_range
        
    def RotateBox(self, img, boxes, angle):
        # rotation matrix
        cos, sin = tf.reshape(tf.math.cos(angle), (1,1)), tf.reshape(tf.math.sin(angle), (1,1))
        matrix_top = tf.concat([cos, -sin], axis = -1)
        matrix_bottom = tf.concat([sin, cos], axis = -1)
        matrix = tf.concat([matrix_top, matrix_bottom], axis = 0)
        
        # center box
        img_shape = tf.dtypes.cast(tf.shape(img), boxes.dtype)
        h = img_shape[:1]
        w = img_shape[1:2]
        half_w = w / 2.
        half_h = h / 2.
        
        d_wh = tf.concat([half_w, half_h], -1)
        d_wh = tf.reshape(d_wh, (1, -1))
        
        xmin, ymin, xmax, ymax = tf.split(boxes[..., :4], [1,1,1,1], -1)

        lt_c = tf.concat([xmin, ymin], -1) - d_wh
        lb_c = tf.concat([xmin, ymax], -1) - d_wh
        rt_c = tf.concat([xmax, ymin], -1) - d_wh
        rb_c = tf.concat([xmax, ymax], -1) - d_wh
        
        # rotate
        r_lt = tf.linalg.matmul(lt_c, matrix) + d_wh
        r_lb = tf.linalg.matmul(lb_c, matrix) + d_wh
        r_rt = tf.linalg.matmul(rt_c, matrix) + d_wh
        r_rb = tf.linalg.matmul(rb_c, matrix) + d_wh
        
        r_lt = tf.expand_dims(r_lt, 0)
        r_lb = tf.expand_dims(r_lb, 0)
        r_rt = tf.expand_dims(r_rt, 0)
        r_rb = tf.expand_dims(r_rb, 0)
        
        n_boxes = tf.concat([r_lt, r_lb, r_rt, r_rb], 0)
        n_boxes_min = tf.reduce_min(n_boxes, 0)
        n_boxes_max = tf.reduce_max(n_boxes, 0) 
        n_boxes = tf.concat([n_boxes_min, n_boxes_max, boxes[..., 4:]], -1)
        
        n_boxes = DropInvalidBox(img, n_boxes)
                
        return n_boxes
    
    def __call__(self, img, boxes):
        angle = tf.random.uniform((), self.angle_range[0], self.angle_range[1])
        h, w, c = tf.split(tf.dtypes.cast(tf.shape(img), boxes.dtype), [1,1,1], 0)
        
        r_img = tfa.image.rotate(img, angle, 'BILINEAR')
        r_boxes = tf.cond(tf.shape(boxes)[0] > 0, lambda : self.RotateBox(img, boxes, angle), lambda : boxes)

        return r_img, r_boxes
        
class ResizeOrCropToInputSize():
    def __init__(self, input_shape, method, crop = True):
        self.input_shape = input_shape # list [h, w]
        self.method = method # string: 'area', 'bicubic', 'bilinear', 'gaussian', 'lanczos3', 'lanczos5', 'mitchellcubic', 'nearest'
        self.crop = crop
        
    def GetResizeParam(self, img):
        ih, iw, ic = tf.split(tf.shape(img), [1,1,1], 0)
        ih = ih[0]
        iw = iw[0]
        ic = ic[0]
        
        if self.crop:
            t_dh = self.input_shape[0] - ih
            t_dw = self.input_shape[0] - iw
            
            case_h_ls = [
                (t_dh == 0, lambda: 0),
                (t_dh > 0 , lambda: tf.random.uniform((), maxval = t_dh, dtype = tf.int32)),
                (t_dh < 0 , lambda: tf.random.uniform((), minval = t_dh, maxval = 0, dtype = tf.int32)),
            ]
            
            dh_head = tf.case(case_h_ls, exclusive = True)
            
            case_w_ls = [
                (t_dw == 0, lambda: 0),
                (t_dw > 0 , lambda: tf.random.uniform((), maxval = t_dw, dtype = tf.int32)),
                (t_dw < 0 , lambda: tf.random.uniform((), minval = t_dw, maxval = 0, dtype = tf.int32)),
            ]
            
            dw_head = tf.case(case_w_ls, exclusive = True)
            
            dh_botton = t_dh - dh_head
            dw_bottom = t_dw - dw_head
            
            return dh_head, dw_head, dh_botton, dw_bottom
            
        else:
            scale = tf.dtypes.cast(tf.math.minimum(self.input_shape[0] / ih, self.input_shape[0] / iw), tf.float32)
            
            nh = tf.dtypes.cast(tf.dtypes.cast(ih, tf.float32) * scale, tf.int32)
            nw = tf.dtypes.cast(tf.dtypes.cast(iw, tf.float32) * scale, tf.int32)
            
            dh = tf.dtypes.cast((self.input_shape[0] - nh) / 2, tf.float32)
            dw = tf.dtypes.cast((self.input_shape[1] - nw) / 2, tf.float32)
            
            return nh, nw, dw, dh, scale
        
    def RandomCropBox(self, boxes, dh_head, dw_head, crop_pad_img):
        xmin, ymin, xmax, ymax = tf.split(boxes[..., :4], [1,1,1,1], -1)
        c_xmin = xmin + tf.dtypes.cast(dw_head, tf.float32)
        c_ymin = ymin + tf.dtypes.cast(dh_head, tf.float32)
        c_xmax = xmax + tf.dtypes.cast(dw_head, tf.float32)
        c_ymax = ymax + tf.dtypes.cast(dh_head, tf.float32)
        
        c_boxes = tf.concat([c_xmin, c_ymin, c_xmax, c_ymax, boxes[..., 4:]], -1)
        
        c_boxes = DropInvalidBox(crop_pad_img, c_boxes)
        return c_boxes
    
    def RandomCrop(self, img, boxes):
        dh_head, dw_head, dh_botton, dw_bottom = self.GetResizeParam(img)
        
        # dh_head >= 0, dw_head >= 0
        def img_pad_hw(img, dh_head, dw_head, dh_botton, dw_bottom):
            dh_head_a = tf.reshape(dh_head, (1,1))
            dw_head_a = tf.reshape(dw_head, (1,1))
            dh_botton_a = tf.reshape(dh_botton, (1,1))
            dw_bottom_a = tf.reshape(dw_bottom, (1,1))
            
            pad_h = tf.concat([dh_head_a, dh_botton_a], -1)
            pad_w = tf.concat([dw_head_a, dw_bottom_a], -1)
            pad_c = tf.zeros((1,2), tf.int32)
            paddings = tf.concat([pad_h, pad_w, pad_c], 0)
            
            pad_img = tf.pad(img, paddings)
            
            return pad_img[:self.input_shape[0], :self.input_shape[1], ...]
        
        # dh_head >= 0, dw_head < 0
        def img_pad_h_crop_w(img, dh_head, dw_head, dh_botton, dw_bottom):
            dh_head_a = tf.reshape(dh_head, (1,1))
            dh_botton_a = tf.reshape(dh_botton, (1,1))
            
            pad_h = tf.concat([dh_head_a, dh_botton_a], -1)
            pad_w = tf.zeros((1,2), tf.int32)
            pad_c = tf.zeros((1,2), tf.int32)
            paddings = tf.concat([pad_h, pad_w, pad_c], 0)
            
            pad_img = tf.pad(img, paddings)
            crop_w = dw_head * -1
            
            return pad_img[:self.input_shape[0], crop_w:(crop_w + self.input_shape[1]), ...]
            
        # dh_head < 0, dw_head >= 0
        def img_crop_h_pad_w(img, dh_head, dw_head, dh_botton, dw_bottom):
            dw_head_a = tf.reshape(dw_head, (1,1))
            dw_bottom_a = tf.reshape(dw_bottom, (1,1))
            
            pad_h = tf.zeros((1,2), tf.int32)
            pad_w = tf.concat([dw_head_a, dw_bottom_a], -1)
            pad_c = tf.zeros((1,2), tf.int32)
            paddings = tf.concat([pad_h, pad_w, pad_c], 0)
            
            pad_img = tf.pad(img, paddings)
            crop_h = dh_head * -1
            
            return pad_img[crop_h:(crop_h + self.input_shape[0]), :self.input_shape[1], ...]
            
        # dh_head  0, dw_head >= 0
        def img_crop_h_crop_w(img, dh_head, dw_head, dh_botton, dw_bottom):
            crop_w = dw_head * -1
            crop_h = dh_head * -1
            
            return img[crop_h:(crop_h + self.input_shape[0]), crop_w:(crop_w + self.input_shape[1]), ...]
        
        
        case_list = [
            (tf.logical_and(dh_head >= 0, dw_head >= 0), lambda: img_pad_hw(img, dh_head, dw_head, dh_botton, dw_bottom)),
            (tf.logical_and(dh_head >= 0, dw_head < 0) , lambda: img_pad_h_crop_w(img, dh_head, dw_head, dh_botton, dw_bottom)),
            (tf.logical_and(dh_head < 0, dw_head >= 0) , lambda: img_crop_h_pad_w(img, dh_head, dw_head, dh_botton, dw_bottom)),
            (tf.logical_and(dh_head < 0, dw_head < 0)  , lambda: img_crop_h_crop_w(img, dh_head, dw_head, dh_botton, dw_bottom))
        ]
        
        crop_pad_img = tf.case(case_list, default=lambda: img, exclusive=True)
        crop_pad_box = tf.cond(tf.shape(boxes)[0] > 0, lambda : self.RandomCropBox(boxes, dh_head, dw_head, crop_pad_img), lambda : boxes)
        
        return crop_pad_img, crop_pad_box
    
    def NormalResizeBox(self, boxes, dw, dh, scale, input_img):
        xmin, ymin, xmax, ymax = tf.split(boxes[..., :4], [1,1,1,1], -1)
        r_xmin = xmin * scale + dw
        r_ymin = ymin * scale + dh
        r_xmax = xmax * scale + dw
        r_ymax = ymax * scale + dh
        
        r_boxes = tf.concat([r_xmin, r_ymin, r_xmax, r_ymax, boxes[..., 4:]], -1)
        r_boxes = DropInvalidBox(input_img, r_boxes)
        return r_boxes
        
    def NormalResize(self, img, boxes = None):
        input_img = tf.image.resize_with_pad(img, self.input_shape[0], self.input_shape[1], self.method)
        
        if boxes is not None:
            nh, nw, dw, dh, scale = self.GetResizeParam(img)
            input_boxes = tf.cond(tf.shape(boxes)[0] > 0, lambda : self.NormalResizeBox(boxes, dw, dh, scale, input_img), lambda : boxes)
            
            return input_img, input_boxes
        
        else:
            return input_img
            
    def __call__(self, img, boxes = None):
        if self.crop:
            return self.RandomCrop(img, boxes)
            
        else:
            return self.NormalResize(img, boxes)
        
class BoxToTensor():
    def __init__(self, input_shape, class_ls, anchor_ls, anchor_mask, reduce_ratio):
        self.input_shape      = input_shape
        self.class_ls         = class_ls
        self.n_class          = len(class_ls)
        self.anchor_ls        = anchor_ls
        self.anchor_mask      = anchor_mask
        self.reduce_ratio     = reduce_ratio
        self.predict_shape_ls = [[input_shape[0] // r, input_shape[1] // r, len(anchor_ls) // 3, 5 + len(class_ls)] for r in reduce_ratio]
        
    def GrideAnchorIdx(self, box_wh):
        # 輸入 boxes, 輸出 [box_idx, best_grid, best_anchor_layer]
        mask_tensor = tf.constant(self.anchor_mask, tf.int32)
        anchor_tensor = tf.constant(self.anchor_ls, tf.float32) # [9, 2]
        anchor_max = anchor_tensor / 2.
        anchor_min = -anchor_max
        anchor_area = anchor_tensor[..., 0] * anchor_tensor[..., 1]
        
        wh = tf.reshape(box_wh, (-1, 1, 2))
        wh_max = wh / 2.
        wh_min = -wh_max
        wh_area = wh[..., 0] * wh[..., 1]
        
        inter_min = tf.math.maximum(wh_min, anchor_min)
        inter_max = tf.math.minimum(wh_max, anchor_max)
        inter_wh = inter_max - inter_min
        inter_area = inter_wh[..., 0] * inter_wh[..., 1]
        
        iou = inter_area / (wh_area + anchor_area - inter_area)
        
        best_anchor_idx = tf.argmax(iou, -1, tf.int32)
        
        best_grid_idx = mask_tensor.shape[0] - 1 - tf.dtypes.cast(best_anchor_idx / mask_tensor.shape[1], tf.int32)
        best_layer_idx = tf.math.floormod(best_anchor_idx, mask_tensor.shape[1])
        
        grid_idx = tf.reshape(best_grid_idx, [-1, 1])
        anchor_idx = tf.reshape(best_layer_idx, [-1, 1])
        
        return grid_idx, anchor_idx
    
    def HWIndex(self, box_xy, grid_idx):
        input_hw = tf.constant(self.input_shape, dtype = tf.float32)
        y_true_hw = tf.constant([shape[:2] for shape in self.predict_shape_ls], dtype = tf.float32)
        box_grid_hw = tf.gather(y_true_hw, tf.reshape(grid_idx, (-1, )))
        hw_dix = tf.dtypes.cast(tf.math.floor(box_xy[...,::-1] / input_hw * box_grid_hw), tf.int32)
        
        return hw_dix
        
    def BoxesMinMaxToXYWH(self, boxes):
        '''
        Box 中的物件位置資料轉換為 xywh 格式
        '''
        box_min = boxes[..., :2]
        box_max = boxes[..., 2:4]
        xy = (box_max + box_min) / 2.
        wh = box_max - box_min

        return tf.concat([xy, wh], -1)
    
    def WithBox(self, boxes, n_box):
        y_true = tf.zeros([3] + self.predict_shape_ls[-1], tf.float32)
        box_xywh = self.BoxesMinMaxToXYWH(boxes) # [n_box, x, y, w, h]
        grid_idx, anchor_idx = self.GrideAnchorIdx(box_xywh[..., 2:]) #[n_box, 2] 2 => gride_idx, anchor_layer_idx
        hw_dix = self.HWIndex(box_xywh[..., :2], grid_idx)
        
        class_idx = tf.dtypes.cast(boxes[..., 5:6], tf.int32) + 5
        xyhwo_indicate = tf.tile(tf.reshape(tf.range(5), (1,-1)), [n_box, 1])
        xyhwoc_indicate = tf.expand_dims(tf.concat([xyhwo_indicate, class_idx], -1), -1)
        
        indices = tf.concat([grid_idx, hw_dix, anchor_idx], -1)
        indices = tf.expand_dims(indices, 1)
        indices = tf.tile(indices, [1,6,1])
        indices = tf.concat([indices, xyhwoc_indicate], -1)
        
        conf_class_value = tf.ones((n_box, 2), box_xywh.dtype)
        stander_xywh = box_xywh / tf.constant(self.input_shape + self.input_shape, box_xywh.dtype)
        updates = tf.concat([stander_xywh, conf_class_value], -1)
        
        y_true = tf.tensor_scatter_nd_update(y_true, indices, updates)
        
        return y_true
        
    def __call__(self, boxes):
        
        n_box = tf.shape(boxes)[0]
        y_true = tf.cond(n_box > 0, lambda: self.WithBox(boxes, n_box), lambda: tf.zeros([3] + self.predict_shape_ls[-1], tf.float32))
        
        y_true_0 = y_true[0, :self.predict_shape_ls[0][0], :self.predict_shape_ls[0][1], ...]
        y_true_1 = y_true[1, :self.predict_shape_ls[1][0], :self.predict_shape_ls[1][1], ...]
        y_true_2 = y_true[2, ...]
        
        return y_true_0, y_true_1, y_true_2 #, indices, updates
