from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
# from datasets import VOCSegmentation, Cityscapes, cityscapes
from torchvision import transforms as T
# from metrics import StreamSegMetrics


import torch
import torch.nn as nn

from PIL import Image
# import matplotlib
# import matplotlib.pyplot as plt
from glob import glob

# Hyun edit
from coco2voc_label import VOC_LABELS
import json
from pycocotools import mask as coco_mask


def predict(input, ckpt, output_path, dataset='voc', gpu_id='0', model='deeplabv3_resnet101', 
            separable_conv=False, output_stride=16, save_val_results_to=None, 
            crop_val=False, val_batch_size=4, crop_size=513):
    """
    Args:
        input: path to image/folder
        ckpt: resume from checkpoint
        output_path: path to output annotation json file
        dataset: Name of training set
        gpu_id: GPU ID
        model: model name
        seprarble_conv: apply separable conv to decoder and aspp
        output_stride: Choice between [8, 16]
        save_val_results: save segmentation results to the specified dir
        crop_val: crop validation (default: False)
        val_batch_size: batch size for validation (default: 4)
        crop_size: image crop size
    """
    BACKGROUND_LABEL = 0
    os.makedirs('torch_home', exist_ok=True)
    os.environ["TORCH_HOME"] = "torch_home"
    
    if dataset == 'voc':
        num_classes = 21
        # decode_fn = VOCSegmentation.decode_target
    elif dataset.lower() == 'cityscapes':
        num_classes = 19
        # decode_fn = Cityscapes.decode_target

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup dataloader
    image_files = []
    if os.path.isdir(input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(input, '**/*.%s'%(ext)), recursive=True)
            if len(files)>0:
                image_files.extend(files)
    elif os.path.isfile(input):
        image_files.append(input)
    
    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[model](num_classes=num_classes, output_stride=output_stride)
    if separable_conv and 'plus' in model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    if ckpt is not None and os.path.isfile(ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        # Hyun Edit : 차라리 없다고 뽑는게 낫지 않나?
        model = nn.DataParallel(model)
        model.to(device)

    #denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if crop_val:
        transform = T.Compose([
                T.Resize(crop_size),
                T.CenterCrop(crop_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    else:
        transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    if save_val_results_to is not None:
        os.makedirs(save_val_results_to, exist_ok=True)
                
    with torch.no_grad():
        model = model.eval()
        
        annotation_gt = {}
        images = []
        annos = []
        anno_id = 0
        for idx, img_path in tqdm(enumerate(sorted(image_files))):
            real_img_name = os.path.basename(img_path)
            
            ext = real_img_name.split('.')[-1]
            # img_name = real_img_name[:-len(ext)-1]
            
            # Hyun Edit: Convert
            img = Image.open(img_path).convert('RGB')
            size = img.size
            
            img_set = {
                'file_name': real_img_name,
                'height': size[1],
                'width': size[0],
                'id': idx
            }
            
            images.append(img_set)
            
            img = transform(img).unsqueeze(0) # To tensor of NCHW
            img = img.to(device)
            
            # pred = model(img)
            # print(pred.max(0))
            # print(pred.max(1)) 
            
            pred = model(img).max(1)[1].cpu().numpy()[0] # HW
                
            # Hyun Edit : To coco format 
            labels = np.unique(pred)
            for label in labels:
                if int(label) != BACKGROUND_LABEL:
                    anno = {}                    
                    anno['iscrowd'] = 1
                    anno['id'] = anno_id
                    anno_id += 1 
                    
                    mask_pred = pred == label
                    seg_rle = coco_mask.encode(np.asfortranarray(np.array(mask_pred, dtype=np.uint8)))
                    
                    # stringfy = seg_rle
                    # stringfy['counts'] = seg_rle['counts'].decode('utf-8')         
                    anno['category_id'] = int(label)
                    anno['image_id'] = idx
                    # both uint32 format...
                    anno['area'] = int(coco_mask.area(seg_rle))
                    anno['bbox'] = coco_mask.toBbox(seg_rle).tolist()
                    seg_rle['counts'] = seg_rle['counts'].decode('utf-8')
                    anno['segmentation'] = seg_rle
                    annos.append(anno)
                
        categories = []
        
        if dataset == 'voc':
            for key, item in VOC_LABELS.items():
                cat = {
                    'id':key,
                    'name': item
                }
                categories.append(cat)
        
        annotation_gt['images'] = images
        annotation_gt['annotations'] = annos                                           
        annotation_gt['categories'] = categories
    
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, 'annotation_gt.json')
    
    with open(output_path, 'w') as writefile:
        json.dump(annotation_gt, writefile)
         