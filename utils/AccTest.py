import numpy as np

from utils.DataProcess import BoxIOU

class SingleImgConfusion():
    def __init__(self, iou_thresh, class_ls):
        self.iou_thresh = iou_thresh
        self.class_ls   = class_ls

    def GetMatchArray(self, true_box, pred_box):
        '''
        true_box: numpy array with shape: [n_box, 6], 6 => xmin, ymin, xmax, ymax, score, class_idx
        pred_box: numpy array with shape: [n_box, 6], 6 => xmin, ymin, xmax, ymax, score, class_idx
        '''
        match_array = [] # [t_box_id, p_box_id, iou, t_class, p_class]
        p_box_idx = np.array(range(pred_box.shape[0])).astype(np.int64)
        t_box_idx = np.array(range(true_box.shape[0])).astype(np.int64)

        for p in range(pred_box.shape[0]):
            for t in range(true_box.shape[0]):
                t_box = true_box[t,:4]
                p_box = pred_box[p,:4]

                t_class = true_box[t,5]
                p_class = pred_box[p,5]

                iou = BoxIOU(t_box, p_box)
                if iou >= self.iou_thresh:
                    match_array.append([t, p, iou, t_class, p_class])

        match_array = np.array(match_array)
        miss_match_array = []
        if match_array.shape[0] > 0:
            match_p_idx = np.unique(match_array[:,1].astype(np.int64))
            miss_match_p_idx = np.setdiff1d(p_box_idx, match_p_idx)
            for p in miss_match_p_idx:
                p_class = pred_box[p,5]
                miss_match_array.append([-1, p, 0, -1, p_class])

            match_t_idx = np.unique(match_array[:,0].astype(np.int64))
            miss_match_t_idx = np.setdiff1d(t_box_idx, match_t_idx)
            for t in miss_match_t_idx:
                t_class = true_box[t,5]
                miss_match_array.append([t, -1, 0, t_class, -1])

            miss_match_array = np.array(miss_match_array)
            if miss_match_array.shape[0] > 0:
                match_array = np.concatenate([match_array, miss_match_array], 0)

        else:
            for p in p_box_idx:
                p_class = pred_box[p,5]
                miss_match_array.append([-1, p, 0, -1, p_class])

            for t in t_box_idx:
                t_class = true_box[t,5]
                miss_match_array.append([t, -1, 0, t_class, -1])

            match_array = np.array(miss_match_array)
        
        return match_array
        
        
    def __call__(self, true_box, pred_box):
        match_array = self.GetMatchArray(true_box, pred_box)
        tp, fp, fn = 0, 0, 0
        
        if match_array.shape[0] > 0:
            # 計算 TP
            tp_mask = match_array[...,3] == match_array[...,4]
            tp_sample = match_array[tp_mask,...]
            tp += tp_sample.shape[0]
            
            # 計算 FP
            fp_mask = np.logical_and(match_array[...,3] != match_array[...,4], match_array[...,1] != -1)
            fp_sample = match_array[fp_mask,...]
            fp += fp_sample.shape[0]
            
            # 計算 FN
            fn_mask = match_array[...,1] == -1
            fn_sample = match_array[fn_mask,...]
            fn += fn_sample.shape[0]
            
        return np.array([tp, fp, fn])