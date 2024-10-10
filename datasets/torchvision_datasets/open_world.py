# partly taken from https://github.com/pytorch/vision/blob/master/torchvision/datasets/voc.py
import functools
import random

import torch
import cv2
import math
import numpy as np
from PIL import Image
import os
import tarfile
import collections
import logging
import copy
from torchvision.datasets import VisionDataset
import itertools

import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg

#OWOD splits
VOC_CLASS_NAMES_COCOFIED = [
    "bigcoal", "lronmesh", "lronbars", "wood"
]


BASE_VOC_CLASS_NAMES = [
    "bigcoal", "lronmesh", "lronbars", "wood"
]

VOC_CLASS_NAMES = [
    "bigcoal", "lronmesh", "lronbars", "wood"
]

T2_CLASS_NAMES = [
    # "bigcoal","water_coal"
]


UNK_CLASS = ["unknown"]

VOC_COCO_CLASS_NAMES = tuple(itertools.chain(VOC_CLASS_NAMES))
print(VOC_COCO_CLASS_NAMES)

class OWDetection(VisionDataset):
    """`OWOD in Pascal VOC format <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
            (default: alphabetic indexing of VOC's 20 classes).
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, required): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self,
                 args,
                 root,
                 years='2007',
                 image_sets='train',
                 transform=None,
                 target_transform=None,
                 transforms=None,
                 no_cats=False,

                 filter_pct=-1):
        super(OWDetection, self).__init__(root, transforms, transform, target_transform)
        self.imgMask = []
        self.images = []
        self.annotations = []
        self.imgids = []
        self.imgid2annotations = {}
        self.image_set = []

        self.CLASS_NAMES = VOC_COCO_CLASS_NAMES
        self.MAX_NUM_OBJECTS = 64
        self.no_cats = no_cats
        self.args = args
        self.conveyor_points = []

        for year, image_set in zip(years, image_sets):

            if year == "2007" and image_set == "test":
                year = "2007-test"
            valid_sets = ["t1_train", "t2_train", "t2_ft","test", "all_task_test"]
            if year == "2007-test":
                valid_sets.append("test")
            # base_dir = DATASET_YEAR_DICT[year]['base_dir']
            voc_root = self.root
            annotation_dir = os.path.join(voc_root, 'Annotations')
            image_dir = os.path.join(voc_root, 'JPEGImages')
            if not os.path.isdir(voc_root):
                raise RuntimeError('Dataset not found or corrupted.' +
                                   ' You can use download=True to download it')
            file_names = self.extract_fns(image_set, voc_root)
            self.image_set.extend(file_names)
            self.images.extend([os.path.join(image_dir, x + ".png") for x in file_names])
            self.annotations.extend([os.path.join(annotation_dir, x + ".xml") for x in file_names])
            self.imgids.extend(self.convert_image_id(x, to_integer=True) for x in file_names)
            self.imgid2annotations.update(dict(zip(self.imgids, self.annotations)))
        if filter_pct > 0:
            num_keep = float(len(self.imgids)) * filter_pct
            keep = np.random.choice(np.arange(len(self.imgids)), size=round(num_keep), replace=False).tolist()
            flt = lambda l: [l[i] for i in keep]
            self.image_set, self.images, self.annotations, self.imgids = map(flt, [self.image_set, self.images,
                                                                                   self.annotations, self.imgids])
        assert (len(self.images) == len(self.annotations) == len(self.imgids))

    @staticmethod
    def convert_image_id(img_id, to_integer=False, to_string=False, prefix='2021'):
        if to_integer:
            return int(prefix + str(img_id))
        if to_string:
            x = str(img_id)
            assert x.startswith(prefix), "Image ID does not start with the expected prefix."
            x = x[len(prefix):]
            return x

    @functools.lru_cache(maxsize=None)
    def load_instances(self, img_id):
        tree = ET.parse(self.imgid2annotations[img_id])
        target = self.parse_voc_xml(tree.getroot())
        image_id = target['annotation']['filename']
        instances = []
        for obj in target['annotation']['object']:
            cls = obj["name"]
            if cls not in VOC_CLASS_NAMES_COCOFIED:
                print(f"Warning: Class '{cls}' not found in VOC_COCO_CLASS_NAMES.")
                continue  # or handle it appropriately

            if cls in VOC_CLASS_NAMES_COCOFIED:
                cls = BASE_VOC_CLASS_NAMES[VOC_CLASS_NAMES_COCOFIED.index(cls)]
            bbox = obj["bndbox"]
            bbox = [float(bbox[x]) for x in ["xmin", "ymin", "xmax", "ymax"]]
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instance = dict(
                category_id=VOC_COCO_CLASS_NAMES.index(cls),
                bbox=bbox,
                area=(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                image_id=img_id
            )
            instances.append(instance)
        return target, instances

    def extract_fns(self, image_set, voc_root):
        splits_dir = os.path.join(voc_root, 'ImageSets/Main')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        return file_names

    def remove_prev_class_and_unk_instances(self, target):
        prev_intro_cls = self.args.PREV_INTRODUCED_CLS
        curr_intro_cls = self.args.CUR_INTRODUCED_CLS
        valid_classes = range(prev_intro_cls, prev_intro_cls + curr_intro_cls)
        entry = copy.copy(target)
        for annotation in copy.copy(entry):
            if annotation["category_id"] not in valid_classes:
                entry.remove(annotation)
        return entry
    def remove_unknown_instances(self, target):
        # For finetune data. Removing the unknown objects...
        prev_intro_cls = self.args.PREV_INTRODUCED_CLS
        curr_intro_cls = self.args.CUR_INTRODUCED_CLS
        valid_classes = range(0, prev_intro_cls+curr_intro_cls)
        entry = copy.copy(target)
        for annotation in copy.copy(entry):
            if annotation["category_id"] not in valid_classes:
                entry.remove(annotation)
        return entry
    def label_known_class_and_unknown(self, target):
        # For test and validation data.
        # Label known instances the corresponding label and unknown instances as unknown.
        prev_intro_cls = self.args.PREV_INTRODUCED_CLS
        curr_intro_cls = self.args.CUR_INTRODUCED_CLS
        total_num_class = self.args.num_classes #81
        known_classes = range(0, prev_intro_cls+curr_intro_cls)
        entry = copy.copy(target)
        for annotation in  copy.copy(entry):
        # for annotation in entry:
            if annotation["category_id"] not in known_classes:
                annotation["category_id"] = total_num_class - 1
        return entry

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        image_set = self.transforms[0]
        img = Image.open(self.images[index]).convert('RGB')
        target, instances = self.load_instances(self.imgids[index])

        conveyor_points1 = [(-400, 750), (500, 0), (850, 0), (1500, 750)]  # 1
        conveyor_points3 = [(-400, 948), (500, 200), (850, 200), (1550, 953)]  # 3
        conveyor_points5 = [(-300, 730), (552, 0), (725, 0), (1400, 730)]  # 5

        if 'train' in image_set:
            instances = self.remove_prev_class_and_unk_instances(instances)
        elif 'test' in image_set:
            instances = self.label_known_class_and_unknown(instances)
        elif 'ft' in image_set:
            instances = self.remove_unknown_instances(instances)
        w, h = map(target['annotation']['size'].get, ['width', 'height'])
        name = target['annotation']['filename']

        self.conveyor_points.append(conveyor_points1)
        conveyor_points = conveyor_points1
        target = dict(
            image_id=torch.tensor([self.imgids[index]], dtype=torch.int64),
            labels=torch.tensor([i['category_id'] for i in instances], dtype=torch.int64),
            area=torch.tensor([i['area'] for i in instances], dtype=torch.float32),
            boxes=torch.as_tensor([i['bbox'] for i in instances], dtype=torch.float32),
            orig_size=torch.as_tensor([int(h), int(w)]),
            size=torch.as_tensor([int(h), int(w)]),
            conveyor_points=torch.tensor([conveyor_points], dtype=torch.float),
            iscrowd=torch.zeros(len(instances), dtype=torch.uint8)
        )

        if self.transforms[-1] is not None:
            img, target = self.transforms[-1](img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == 'annotation':
                def_dic['object'] = [def_dic['object']]
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict


def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), "r") as tar:
        tar.extractall(path=root)

def getRotatedImg(angle, img):
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    rows, cols = img_cv.shape[:2]
    a, b = cols / 2, rows / 2
    M = cv2.getRotationMatrix2D((a, b), angle, 1)
    rotated_img_cv = cv2.warpAffine(img_cv, M, (cols, rows))
    return rotated_img_cv, a, b, M


def getRotatedPoints(angle_rad, a, b, points):
    rotated_points = [
        ((x - a) * math.cos(angle_rad) + (y - b) * math.sin(angle_rad) + a,
         -(x - a) * math.sin(angle_rad) + (y - b) * math.cos(angle_rad) + b)
        for x, y in points
    ]
    return rotated_points


def getRotatedAnno(angle_rad, a, b, instances, conveyor_points):
    rotated_instances = []
    for instance in instances:
        xmin, ymin, xmax, ymax = instance['bbox']
        points = [(xmin, ymin), (xmax, ymax), (xmin, ymax), (xmax, ymin)]
        rotated_points = getRotatedPoints(angle_rad, a, b, points)
        Xs, Ys = zip(*rotated_points)
        X_MIN, X_MAX = min(Xs), max(Xs)
        Y_MIN, Y_MAX = min(Ys), max(Ys)

        rotated_instance = instance.copy()
        rotated_instance['bbox'] = [X_MIN, Y_MIN, X_MAX, Y_MAX]
        rotated_instances.append(rotated_instance)

    rotated_conveyor_points = getRotatedPoints(angle_rad, a, b, conveyor_points)
    return rotated_instances, rotated_conveyor_points


def rotate(angle, img, instances, conveyor_points):
    angle_rad = angle * math.pi / 180.0
    rotated_img_cv, a, b, M = getRotatedImg(angle, img)
    rotated_instances, rotated_conveyor_points = getRotatedAnno(angle_rad, a, b, instances, conveyor_points)
    return rotated_instances, rotated_conveyor_points
