import tensorflow as tf
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, LeakyReLU, Add, UpSampling2D
from tensorflow.keras.regularizers import l2

class HeadConv(tf.keras.Model):
    def __init__(self, filters, alpha, l2_decay = 0,  *args, **kwargs):
        super(HeadConv, self).__init__(*args, **kwargs)
        self.conv = Conv2D(filters, 3, (1,1), 'same', use_bias = False, kernel_regularizer = l2(l=l2_decay))
        self.bn   = BatchNormalization(trainable=True)
        self.act  = LeakyReLU(alpha)
        
    def call(self, input_tensor, training = False):
        x = self.conv(input_tensor)
        x = self.bn(x, training = training)
        x = self.act(x)
        
        return x
    
    def get_config(self):
        return {'units': self.units}
        
class DarknetModule(tf.keras.Model):
    def __init__(self, filters, alpha, l2_decay = 0, residual_loops = 1,  *args, **kwargs):
        super(DarknetModule, self).__init__(*args, **kwargs)
        
        filters_2 = filters * 2
        
        # 包含 Zero Padding 且 stride = (2, 2 的卷積)
        self.zero_pad = ZeroPadding2D(((1,0),(1,0)))
        self.zero_pad_conv = Conv2D(filters_2, 3, (2,2), 'valid', use_bias = False, kernel_regularizer = l2(l=l2_decay))
        self.zero_pad_batch_norm = BatchNormalization(trainable=True)
        self.zero_pad_act = LeakyReLU(alpha)
        
        self.res_block_ls = []
        
        for i in range(residual_loops):
            tmp_dic = {}
            tmp_dic["conv_1"] = Conv2D(filters, 1, (1,1), 'same', use_bias = False, kernel_regularizer = l2(l=l2_decay))
            tmp_dic["batch_norm_1"] = BatchNormalization(trainable=True)
            tmp_dic["act_1"] = LeakyReLU(alpha)
            
            tmp_dic["conv_2"] = Conv2D(filters_2, 3, (1,1), 'same', use_bias = False, kernel_regularizer = l2(l=l2_decay))
            tmp_dic["batch_norm_2"] = BatchNormalization(trainable=True)
            tmp_dic["act_2"] = LeakyReLU(alpha)
            
            tmp_dic["add"] = Add()
            self.res_block_ls.append(tmp_dic)
            
    def call(self, input_tensor, training=False):
        x = self.zero_pad(input_tensor)
        x = self.zero_pad_conv(x)
        x = self.zero_pad_batch_norm(x, training = training)
        x = self.zero_pad_act(x)
        
        for res_block in self.res_block_ls:
            conv = res_block["conv_1"](x)
            conv = res_block["batch_norm_1"](conv, training = training)
            conv = res_block["act_1"](conv)
            
            conv = res_block["conv_2"](conv)
            conv = res_block["batch_norm_2"](conv, training = training)
            conv = res_block["act_2"](conv)
            
            x = res_block["add"]([conv, x])
            
        return x

class YoloModule(tf.keras.Model):
    def __init__(self, filters, filters_out, alpha, l2_decay = 0, end = False,  *args, **kwargs):
        super(YoloModule, self).__init__( *args, **kwargs)
        
        filters_2 = filters * 2
        filters_half = filters // 2
        
        self.end = end
        
        self.conv_11 = Conv2D(filters, 1, (1,1), 'same', use_bias = False, kernel_regularizer = l2(l=l2_decay))
        self.bn_11   = BatchNormalization(trainable=True)
        self.act_11  = LeakyReLU(alpha)
        
        self.conv_12 = Conv2D(filters_2, 3, (1,1), 'same', use_bias = False, kernel_regularizer = l2(l=l2_decay))
        self.bn_12   = BatchNormalization(trainable=True)
        self.act_12  = LeakyReLU(alpha)
        
        self.conv_21 = Conv2D(filters, 1, (1,1), 'same', use_bias = False, kernel_regularizer = l2(l=l2_decay))
        self.bn_21   = BatchNormalization(trainable=True)
        self.act_21  = LeakyReLU(alpha)
        
        self.conv_22 = Conv2D(filters_2, 3, (1,1), 'same', use_bias = False, kernel_regularizer = l2(l=l2_decay))
        self.bn_22   = BatchNormalization(trainable=True)
        self.act_22  = LeakyReLU(alpha)
        
        self.conv_31 = Conv2D(filters, 1, (1,1), 'same', use_bias = False, kernel_regularizer = l2(l=l2_decay))
        self.bn_31   = BatchNormalization(trainable=True)
        self.act_31  = LeakyReLU(alpha)
        
        self.conv_out_1 = Conv2D(filters_2, 3, (1,1), 'same', use_bias = False, kernel_regularizer = l2(l=l2_decay))
        self.bn_out_1   = BatchNormalization(trainable=True)
        self.act_out_1  = LeakyReLU(alpha)
        
        self.conv_out2 = Conv2D(filters_out, 1, (1,1), 'same', use_bias = True, kernel_regularizer = l2(l=l2_decay))
        
        if not self.end:
            self.conv_up = Conv2D(filters_half, 1, (1,1), 'same', use_bias = False, kernel_regularizer = l2(l=l2_decay))
            self.bn_up   = BatchNormalization(trainable=True)
            self.act_up  = LeakyReLU(alpha)

            self.upsample = UpSampling2D()
        
    def call(self, input_tensor, training=False):
        x = self.conv_11(input_tensor)
        x = self.bn_11(x, training = training)
        x = self.act_11(x)
        
        x = self.conv_12(x)
        x = self.bn_12(x, training = training)
        x = self.act_12(x)
        
        x = self.conv_21(x)
        x = self.bn_21(x, training = training)
        x = self.act_21(x)
        
        x = self.conv_22(x)
        x = self.bn_22(x, training = training)
        x = self.act_22(x)
        
        x = self.conv_31(x)
        x = self.bn_31(x, training = training)
        x = self.act_31(x)
        
        y = self.conv_out_1(x)
        y = self.bn_out_1(y, training = training)
        y = self.act_out_1(y)
        
        y = self.conv_out2(y)
        
        if not self.end:
            x = self.conv_up(x)
            x = self.bn_up(x, training = training)
            x = self.act_up(x)
            
            x = self.upsample(x)
            
        return x, y
    
class ScaleOutputBox():
    def __init__(self, anchors, x_shape_wh, class_method = 'sigmoid', training = True):
        self.anchors      = anchors
        self.x_shape_wh   = x_shape_wh
        self.training     = training
        self.class_method = class_method # "sigmoid" or "softmax"

    def ComputCxy(self, h, w):
        # x y offset
        cx  = tf.tile(tf.reshape(tf.range(w), [1, -1, 1, 1]), [h, 1, 1, 1])
        cy  = tf.tile(tf.reshape(tf.range(h), [-1, 1, 1, 1]), [1, w, 1, 1])
        cxy = tf.concat([cx, cy], axis = -1)

        return cxy

    def RescaleOutput(self, y, y_shape, y_shape_wh_f, x_shape_wh_f, anchors_f):        
        h, w, a, c = y_shape[1], y_shape[2], y_shape[3], y_shape[4]
        cxy   = self.ComputCxy(h, w)
        cxy_f = tf.dtypes.cast(cxy, y.dtype)
        
        p_xy    = (tf.math.sigmoid(y[..., :2]) + cxy_f) / y_shape_wh_f # predict xy
        p_wh    = tf.math.exp(y[..., 2:4]) * anchors_f / x_shape_wh_f  # predict wh 
        p_conf  = tf.math.sigmoid(y[..., 4:5])                         # predict conf

        if self.class_method == "softmax":
            p_class = tf.math.softmax(y[..., 5:])                          # predict class
        else: 
            p_class = tf.math.sigmoid(y[..., 5:])
            
        output_box = tf.concat([p_xy, p_wh, p_conf, p_class], -1)
        
        if self.training:
            return output_box, cxy_f
        
        else:
            return tf.reshape(output_box, (-1, h * w * a, c)) # 弄成 NMS 的形狀 (batch, nbox, 1, 5 + n_classes)
                
    def RescaleToNmsStyle(self, output_box):
        '''
        X, Y, W, H => Y_min, X_min, Y_max, X_max
        output_box: shape => (batch, nbox, 1, 4 + 1 + n_classes)
        '''
        xy = output_box[..., :2]
        wh = output_box[..., 2:4]
        wh_half = wh / 2.
        scores = output_box[..., 4:5] * output_box[..., 5:] #[batch, n_box, n_classes]
        
        
        boxes_min = (xy - wh_half)[..., ::-1]
        boxes_max = (xy + wh_half)[..., ::-1]
        boxes = tf.concat([boxes_min, boxes_max], axis = -1) #[batch, n_box, 4]
        
        return boxes, scores
    
    def __call__(self, y, orign_img_hw = None):
        y_shape = tf.shape(y)
        y_shape_wh = y_shape[1:3][::-1]
        y_shape_wh_f = tf.dtypes.cast(y_shape_wh      , y.dtype)
        x_shape_wh_f = tf.dtypes.cast(self.x_shape_wh , y.dtype)
        anchors_f    = tf.dtypes.cast(self.anchors    , y.dtype)
        
        if self.training:
            output_box, cxy_f = self.RescaleOutput(y, y_shape, y_shape_wh_f, x_shape_wh_f, anchors_f)
            return output_box, cxy_f, anchors_f, x_shape_wh_f, y_shape_wh_f
        
        else:
            output_box = self.RescaleOutput(y, y_shape, y_shape_wh_f, x_shape_wh_f, anchors_f)
            p_box, p_score = self.RescaleToNmsStyle(output_box)
            
            return p_box, p_score
            
class ComputeIgnoreMask():
    def __init__(self, iou_thresh):
        self.iou_thresh = iou_thresh
        
    def IgnoreMaskIou(self, p_i_xy, p_i_wh, g_i_xy, g_i_wh):
        '''
        p_i_xy, p_i_wh shape => (H, W, num_anchors, 1, 2)
        g_i_xy, g_i_wh shape => (1, num_obj, 2)
        '''
        p_i_wh_half = p_i_wh / 2

        p_i_min  = p_i_xy - p_i_wh_half
        p_i_max  = p_i_xy + p_i_wh_half
        p_i_area = p_i_wh[..., 0] * p_i_wh[..., 1]

        g_i_wh_half = g_i_wh / 2

        g_i_min  = g_i_xy - g_i_wh_half
        g_i_max  = g_i_xy + g_i_wh_half
        g_i_area = g_i_wh[..., 0] * g_i_wh[..., 1]

        inter_min  = tf.math.maximum(p_i_min, g_i_min)
        inter_max  = tf.math.minimum(p_i_max, g_i_max)
        inter_wh   = tf.math.maximum(inter_max - inter_min, 0)
        inter_area = inter_wh[..., 0] * inter_wh[..., 1] # (H, W, num_anchors, num_obj)

        union_area = p_i_area + g_i_area - inter_area
        iou = inter_area / union_area

        return iou
    
    def GetSampleIgnoreMask(self, spg_box):
        '''
        spg_box = tf.concat([sp_xy, sp_wh, sg_xy, sg_wh, sg_conf] axis = -1)
        '''
        sp_box, sg_box, sg_conf = tf.split(spg_box, [4,4,1], axis = -1)
        
        sp_box = tf.expand_dims(sp_box, -2)               # (H, W, A, 1, 4)
        sp_xy, sp_wh = tf.split(sp_box, [2,2], axis = -1) # (H, W, A, 1, 2)
        
        strue_box = tf.boolean_mask(sg_box, sg_conf[...,0])  # (num_box, 4)
        strue_box = tf.expand_dims(strue_box, 0)             # (1, num_box, 4)
        sg_xy, sg_wh = tf.split(strue_box, [2,2], axis = -1) # (1, num_box, 2)
        
        iou = self.IgnoreMaskIou(sp_xy, sp_wh, sg_xy, sg_wh)
        best_iou = tf.math.reduce_max(iou, -1)                         # (H, W, anchors)
        mask = tf.dtypes.cast(best_iou < self.iou_thresh, sg_xy.dtype) # (H, W, anchors)
        
        return mask
    
    def __call__(self, y_true, p_box):
        '''
        p_xy, p_wh, g_xy, g_wh => (batch, H, W, Anchor, 2)
        g_conf  => (batch, H, W, Anchor, 1)
        '''
        pg_box = tf.concat([p_box[..., :4], y_true[..., :5]], axis = -1)
        ignore_mask = tf.map_fn(self.GetSampleIgnoreMask, pg_box)
        
        return ignore_mask

class YoloLoss():
    def __init__(self, anchors, x_shape_wh, iou_thresh, class_method = 'sigmoid'):
        self.anchors      = anchors
        self.x_shape_wh   = x_shape_wh
        self.iou_thresh   = iou_thresh
        self.class_method = class_method
        
    def LossXY(self, g_xy, y_xy, pos_sample, box_scale, y_shape_wh_f, cxy_f):
        sig_xy = (g_xy * y_shape_wh_f) - cxy_f
        xy_ce = pos_sample * box_scale * tf.keras.losses.binary_crossentropy(sig_xy, y_xy, from_logits=True) # cross entropy
        
        return tf.math.reduce_sum(xy_ce)
    
    def LossWH(self, g_wh, y_wh, g_conf, pos_sample, box_scale, x_shape_wh_f, anchors_f):
        exp_wh = g_wh * x_shape_wh_f / anchors_f # g_wh to exp scale
        exp_wh = tf.where(tf.dtypes.cast(g_conf, tf.bool), exp_wh, tf.ones_like(exp_wh)) # change 0 to 1 to avoid -inf when turn it to log scale
        log_wh = tf.math.log(exp_wh) # log_wh = log(exp_wh)
        
        wh_se = pos_sample * box_scale * tf.keras.losses.MSE(log_wh, y_wh) # mean squer error
        
        return tf.math.reduce_sum(wh_se)
    
    def LossConf(self, g_conf, y_conf, y_true, p_box, pos_sample, no_sample):
        ignore_mask = ComputeIgnoreMask(self.iou_thresh)(y_true, p_box)
        neg_sample = no_sample * ignore_mask

        conf_ce = tf.keras.losses.binary_crossentropy(g_conf, y_conf, from_logits=True) # conf ce
        
        pos_ce  = pos_sample * conf_ce # pos conf loss
        neg_ce  = neg_sample * conf_ce # neg conf loss
        obj_ce  = pos_ce + neg_ce      # conf loss
        
        return tf.math.reduce_sum(obj_ce)
        
    def LossClass(self, g_class, y_class, pos_sample):
        #class_ce  = tf.keras.losses.categorical_crossentropy(g_class, p_class) * pos_sample
        class_ce  = tf.keras.losses.binary_crossentropy(g_class, y_class, from_logits=True) * pos_sample
        
        return tf.math.reduce_sum(class_ce)
        
    def __call__(self, y_true, y_pred, sample_weight = None):
        batch_size = tf.dtypes.cast(tf.shape(y_true)[0], y_true.dtype)
        p_box, cxy_f, anchors_f, x_shape_wh_f, y_shape_wh_f = ScaleOutputBox(self.anchors, self.x_shape_wh, self.class_method, True)(y_pred)
        
        p_xy, p_wh, p_conf, p_class = p_box[..., :2],  p_box[..., 2:4],  p_box[..., 4:5],  p_box[..., 5:]
        y_xy, y_wh, y_conf, y_class = y_pred[..., :2], y_pred[..., 2:4], y_pred[..., 4:5], y_pred[..., 5:]
        g_xy, g_wh, g_conf, g_class = y_true[..., :2], y_true[..., 2:4], y_true[..., 4:5], y_true[..., 5:]
        
        
        # y_xy, y_wh, y_conf, y_class = y_pred[..., :2], y_pred[..., 2:4], y_pred[..., 4:5], y_pred[..., 5:]
        y_wh = y_pred[..., 2:4]
        pos_sample = g_conf[..., 0] # positive sample
        no_sample  = 1. - pos_sample
        
        box_scale = 2. - (g_wh[..., 0] * g_wh[..., 1]) # box scale
        
        xy_loss    = self.LossXY(g_xy, y_xy, pos_sample, box_scale, y_shape_wh_f, cxy_f) # compute xy loss
        wh_loss    = self.LossWH(g_wh, y_wh, g_conf, pos_sample, box_scale, x_shape_wh_f, anchors_f) # compute wh loss
        obj_loss   = self.LossConf(g_conf, y_conf, y_true, p_box, pos_sample, no_sample) # compute conf loss
        class_loss = self.LossClass(g_class, y_class, pos_sample) # class loss
        
        return (xy_loss + wh_loss + obj_loss + class_loss) / batch_size
        
    
class NMS():
    def __init__(self, iou_thresh, score_thresh, input_shape, max_output_size_per_class = 100, max_total_size = 100):
        self.iou_thresh   = iou_thresh[0,0]   # tensor (1,1)
        self.score_thresh = score_thresh[0,0] # tensor (1,1)
        self.input_shape  = input_shape # list (h, w)
        
        self.max_output_size_per_class = max_output_size_per_class
        self.max_total_size            = max_total_size
        
    def __call__(self, boxes, scores, orign_img_hw):
        '''
        輸入參數: 
        boxes:        [batch_size, num_boxes, 4] ,scale => 0 ~ 1
        socre:        [batch_size, num_boxes, num_classes]
        orign_img_hw: [batch_size, 2], 2 => h, w

        nms 回傳參數:
        nmsed_boxes:      [batch_size, num_max_boxes, 4], 4 => ymin, xmin, ymax, xmax
        nmsed_scores:     [batch_size, num_max_boxes]
        nmsed_classes:    [batch_size, num_max_boxes]   , 每個 box 的 class 號碼
        valid_detections: [batch_size]                  , batch 中 nmsed_boxes 到第幾個 boxes 為真的有辨識到物件的 box
        '''
        intput_hw = tf.expand_dims(tf.constant(self.input_shape, tf.float32), 0) # [1, 2]
        boxes = boxes * tf.tile(intput_hw, [1, 2]) # return to the input shape size

        scale = tf.math.reduce_min(intput_hw / orign_img_hw, -1, keepdims = True) # [batch_size, 1]
        
        n_hw = orign_img_hw * scale      # [batch_size, 2]
        d_hw = (intput_hw - n_hw) / 2
        d_hw = tf.expand_dims(d_hw, -2)   # [batch_size, 1, 2]
        scale = tf.expand_dims(scale, -2) # [batch_size, 1, 1]

        orign_img_hw = tf.tile(tf.expand_dims(orign_img_hw, -2), [1, 1, 2])
        correction_boxes = (boxes - tf.tile(d_hw, [1, 1, 2])) / scale # return to the original image scale
        correction_norm_boxes = correction_boxes / orign_img_hw # normalize to interval [0, 1]
        correction_norm_boxes = tf.expand_dims(correction_norm_boxes, -2) # [batch_size, num_boxes, 1, 4]

        nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = tf.image.combined_non_max_suppression(
            correction_norm_boxes,
            scores,
            self.max_output_size_per_class,
            self.max_total_size,
            iou_threshold = self.iou_thresh,
            score_threshold = self.score_thresh,
            pad_per_class = False,
            clip_boxes = True
        )

        nmsed_boxes = nmsed_boxes * orign_img_hw

        return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections
    
class GradientCallback(tf.keras.callbacks.Callback):
    console = True

    def on_epoch_end(self, epoch, logs=None):
        weights = [w for w in self.model.trainable_weights if 'dense' in w.name and 'bias' in w.name]
        loss = self.model.total_loss
        optimizer = self.model.optimizer
        gradients = optimizer.get_gradients(loss, weights)
        for t in gradients:
            if self.console:
                print('Tensor: {}'.format(t.name))
                print('{}\n'.format(K.get_value(t)[:10]))
            else:
                tf.summary.histogram(t.name, data=t)