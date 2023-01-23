import json
import os
import numpy as np
from metrics import StreamSegMetrics
from pycocotools import mask as coco_mask

# target_anno_path = 'hyun/samples/voc_seg_230114_dev-scale_7.5-ca_step_0.0_0.5/annotations.json'
# gt_anno_path = 'annotation_gt.json'

def evaluate_iou(target_anno_path, gt_anno_path, data_type='voc'):
    """
    Args:
        target_anno_path (string): path to diffaug annotation.
        gt_anno_path (string): path to ground truth annnotation (annotated from DeepLabV3).
        data_type (str, optional): Dataset name. Defaults to 'voc'.

    Returns:
        result: dict {
            "Overall Acc": acc,
            "Mean Acc": acc_cls,
            "FreqW Acc": fwavacc,
            "Mean IoU": mean_iu,
            "Class IoU": cls_iu,
        }
    """

    target_anno = json.load(open(target_anno_path, 'r'))
    gt_anno = json.load(open(gt_anno_path, 'r'))

    if data_type == 'voc':
        num_classes = 21
    elif data_type == 'cityscapes':
        num_classes = 19
    
    metrics = StreamSegMetrics(num_classes)
    # metric에서 어차피 np니까 비교하고, decode써서 np 만들자

    true_labels = gt_anno['annotations']
    pred_labels = target_anno['annotations']

    true_images = gt_anno['images']
    pred_images = target_anno['images']
    pred_ids = []

    true_annos = {}
    for ti in true_images:
        image_id = ti['id']
        img_shape = np.zeros((ti['width'], ti['height']))
        for tl in true_labels:  
            if int(tl['image_id'])  == int(image_id):
                label = tl['category_id']
                nshape = tl['segmentation']
                # first encode string to binary than to numpy
                nshape['counts'] = nshape['counts'].encode('utf-8')

                numpy_shape = coco_mask.decode(nshape)
                img_shape += numpy_shape*label
                    
        true_annos[image_id] = img_shape
        
    pred_annos = {}
    for ti in pred_images:
        image_id = ti['id']
        pred_ids.append(image_id)
        
        img_shape = np.zeros((ti['width'], ti['height']))
        
        for tl in pred_labels:
            if tl['image_id']  == image_id:
                label = tl['category_id']
                nshape = tl['segmentation']
                # first encode string to binary than to numpy
                nshape['counts'] = nshape['counts'].encode('utf-8')

                numpy_shape = coco_mask.decode(nshape)
                img_shape += numpy_shape*label

        pred_annos[image_id] = img_shape

    pred_arrays = []
    true_arrays = []

    for k in sorted(pred_ids):
        pred_arrays.append(pred_annos[k])
        true_arrays.append(true_annos[k])
        
    pred_arrays = np.array(pred_arrays)
    true_arrays = np.array(true_arrays)

    metrics.update_hyun(true_arrays.astype(int), pred_arrays.astype(int))
    result = metrics.get_results_hyun()
    
    return result
