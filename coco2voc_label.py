import argparse, json
import os

VOC_LABELS= {
    1:'aeroplane',
    2:'bicycle',
    3:'bird',
    4:'boat',
    5:'bottle',
    6:'bus',
    7:'car',
    8:'cat',
    9:'chair',
    10:'cow',
    11:'diningtable',
    12:'dog',
    13:'horse',
    14:'motorbike',
    15:'person',
    16:'pottedplant',
    17:'sheep',
    18:'sofa',
    19:'train',
    20:'tvmonitor', 
}

COCO_LABELS = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}

VOC2COCO = {
    1: 5,
    2: 2,
    3: 16,
    4: 9,
    5: 44,
    6: 6,
    7: 3,
    8: 17,
    9: 62,
    10: 21,
    11: 67,
    12: 18,
    13: 19,
    14: 4,
    15: 1,
    16: 64,
    17: 20,
    18: 63,
    19: 7,
    20: 72,
}

COCO2VOC = {}
for key, item in VOC2COCO.items():
    COCO2VOC[item] = key

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    content = json.load(open(args.anno_file, 'r'))
    
    new_annos = []
    image_ids = set()
    
    for anno in content['annotations']:
        if anno['category_id'] in COCO2VOC.keys():
            anno['category_id'] = COCO2VOC[anno['category_id']]
            new_annos.append(anno)
            image_ids.add(anno['image_id'])
    
    new_imgs = []
    for img in content['images']:
        if img['id'] in image_ids:
            new_imgs.append(img)
    
    new_cats = []
    for cat in content['categories']:
        if cat['id'] in COCO2VOC.keys():
            cat['id'] = COCO2VOC[cat['id']]
            cat['name'] = VOC_LABELS[COCO2VOC[cat['id']]]
            new_cats.append(cat)
    
    new_content= {
        'images': new_imgs,
        'annotations': new_annos,
        'categories': new_cats,
    }
    file_name = os.path.basename(args.anno_file)
    
    json.dump(new_content, open(os.path.join(args.output_dir, f'voc_label_{file_name}'), 'w'))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_file", help="annotation file for object instance/keypoint")
    # parser.add_argument("--type", type=str, help="object instance or keypoint", choices=['instance', 'keypoint'])
    parser.add_argument("--output_dir", help="output directory for voc annotation xml file")
    args = parser.parse_args()
    main(args)