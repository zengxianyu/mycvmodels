import os
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils import data
from torchvision import transforms
import pdb
import random
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xml.etree.ElementTree as ET
import cv2


VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')
#
#
# def uint82bin(n, count=8):
#     """returns the binary of integer n, count refers to amount of bits"""
#     return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])
#
#
# def labelcolormap(N):
#     cmap = np.zeros((N, 3), dtype=np.uint8)
#     for i in range(N):
#         r = 0
#         g = 0
#         b = 0
#         id = i
#         for j in range(7):
#             str_id = uint82bin(id)
#             r = r ^ (np.uint8(str_id[-1]) << (7 - j))
#             g = g ^ (np.uint8(str_id[-2]) << (7 - j))
#             b = b ^ (np.uint8(str_id[-3]) << (7 - j))
#             id = id >> 3
#         cmap[i, 0] = r
#         cmap[i, 1] = g
#         cmap[i, 2] = b
#     return cmap
#
#
# index2color = labelcolormap(21)
# index2color = [tuple(hh/255.0) for hh in index2color]
# index2name = ['background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
#               'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'potted plant',
#               'sheep', 'sofa', 'train', 'tv/monitor']



def crop_img_boxes(img, boxes, crop):
    """
    randomly crop a path from img.
    :param img: PIL image
    :param boxes: N*4 np array
    :param crop: float
    :return: cropped img and boxes
    """
    w, h = img.size
    th, tw = int(crop * h), int(crop * w)
    if w == tw and h == th:
        return 0, 0, h, w
    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)
    img = img.crop((j, i, j + tw, i + th))
    boxes[:, 0] -= j
    boxes[:, 2] -= j
    boxes[:, 1] -= i
    boxes[:, 3] -= i
    boxes[boxes[:, 0] < 0, 0] = 0
    boxes[boxes[:, 1] < 0, 1] = 0
    boxes[boxes[:, 2] > tw - 1, 2] = tw - 1
    boxes[boxes[:, 3] > th - 1, 3] = th - 1
    return img, boxes


def resize_img_boxes(img, boxes, imsize):
    """
    resize img to a square of size imsize
    :param img: PIL Image
    :param boxes: N*4 np array
    :param imsize: int
    :return: img, boxes
    """
    W, H = img.size
    img = img.resize((imsize, imsize))
    boxes[:, 0] *= (float(imsize) / W)
    boxes[:, 2] *= (float(imsize) / W)
    boxes[:, 1] *= (float(imsize) / H)
    boxes[:, 3] *= (float(imsize) / H)
    return img, boxes


def hflip_img_boxes(img, boxes):
    img = ImageOps.mirror(img)
    W, H = img.size
    _boxes = boxes.copy()
    _boxes[:, 0] = W - boxes[:, 2]
    _boxes[:, 2] = W - boxes[:, 0]
    return img, _boxes


class VOCDet(data.Dataset):
    def __init__(self, img_dir, gt_dir, split_file, flip=True, crop=0.9, size=256, mean=None, std=None):
        super(VOCDet, self).__init__()
        self.crop = crop
        self.flip = flip
        self.size = size
        self.mean = mean
        self.std = std
        with open(split_file, 'r') as f:
            self.names = f.read().split('\n')[:-1]
        self.img_files = [os.path.join(img_dir, name) for name in self.names]
        self.ann_files = [os.path.join(gt_dir, name) for name in self.names]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        img = Image.open(self.img_files[index]+'.jpg').convert('RGB')
        ann = ET.parse(self.ann_files[index]+'.xml')
        boxes = []
        labels = []
        diffs = []
        for ann_obj in ann.findall('object'):
            boxes += [[int(ann_obj.find('bndbox').find(tag).text)-1
                       for tag in ['xmin', 'ymin', 'xmax', 'ymax']]]
            labels += [VOC_BBOX_LABEL_NAMES.index(ann_obj.find('name').text)]
            diffs += [int(ann_obj.find('difficult').text)]
        boxes = np.array(boxes, dtype=np.float)
        labels = np.array(labels, dtype=np.int32)
        diffs = np.array(diffs, dtype=np.int32)
        # ================random crop==================
        if self.crop is not None:
            img, boxes = crop_img_boxes(img, boxes, self.crop)
        # ================resize=======================
        img, boxes = resize_img_boxes(img, boxes, self.size)
        # ===============random flip===================
        if self.flip and random.randint(0, 1):
            img, boxes = hflip_img_boxes(img, boxes)
        # ================normalize====================
        img = np.array(img)
        img = img.astype(np.float32)
        if self.mean is not None:
            img -= self.mean
        if self.std is not None:
            img /= self.std
        # boxes = boxes[:, [1, 0, 3, 2]]  # ymin, xmin, ymax, xmax
        img = np.transpose(img, (2, 0, 1))
        img = torch.Tensor(img)
        boxes = torch.Tensor(boxes)
        labels = torch.LongTensor(labels)
        diffs = torch.LongTensor(diffs)
        return img, boxes, labels, diffs


if __name__ == "__main__":
    sb=VOCDet('/home/zeng/data/datasets/segmentation_Dataset/VOCdevkit/VOC2012/JPEGImages',
              '/home/zeng/data/datasets/segmentation_Dataset/VOCdevkit/VOC2012/Annotations',
              '/home/zeng/data/datasets/segmentation_Dataset/VOCdevkit/VOC2012/ImageSets/Main/train.txt')
    for img, boxes, labels, diffs in sb:
        img = np.transpose(img, (1, 2, 0))
        fig, ax = plt.subplots(1)
        ax.imshow(img/255)
        for box in boxes:
            rect = patches.Rectangle(
                (box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        # for roi in rois[:10]:
        #     rect = patches.Rectangle(
        #         (roi[0], roi[1]), roi[2]-roi[0], roi[3]-roi[1], linewidth=1, edgecolor='r', facecolor='none')
        #     ax.add_patch(rect)
        plt.show()
    pdb.set_trace()