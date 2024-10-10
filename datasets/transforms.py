# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Transforms and data augmentation for both image + bbox.
"""
import random
from PIL import Image
import numpy as np
import cv2

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.ops import box_convert
from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate


def crop(image, target, region):
    cropped_image = F.crop(image, *region)
    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")
    if "conveyor_points" in target:
        conveyor_points = target["conveyor_points"]
        cropped_conveyor_points = conveyor_points - torch.as_tensor([j, i])
        target["conveyor_points"] = cropped_conveyor_points
    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def augment_hsv_pil(im_pil,hgain, sgain, vgain):

    im = np.array(im_pil)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        result_im = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
        result_im = cv2.cvtColor(result_im, cv2.COLOR_BGR2RGB)

    result_im_pil = Image.fromarray(result_im)
    return result_im_pil

def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes
    if "conveyor_points" in target:
        new_x = w - target["conveyor_points"][..., 0]
        new_y = target["conveyor_points"][..., 1]
        target["conveyor_points"] = torch.stack((new_x, new_y), dim=-1)

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes
    if "conveyor_points" in target:
        conveyor_points = target["conveyor_points"]
        scaled_conveyor_points = conveyor_points * torch.tensor([ratio_width, ratio_height])
        target["conveyor_points"] = scaled_conveyor_points

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def random_perspective_pil(im, targets=None, degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                           border=(0, 0)):
    # Convert PIL image to numpy array
    im_np = np.array(im)

    height = im_np.shape[0] + border[0] * 2
    width = im_np.shape[1] + border[1] * 2

    # Matrix transformations using PyTorch
    C = torch.eye(3)
    C[0, 2] = -width / 2
    C[1, 2] = -height / 2

    P = torch.eye(3)
    P[2, 0] = torch.FloatTensor([random.uniform(-perspective, perspective)])
    P[2, 1] = torch.FloatTensor([random.uniform(-perspective, perspective)])

    R = torch.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = torch.FloatTensor(cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s))

    S = torch.eye(3)
    S[0, 1] = torch.tan(torch.FloatTensor([random.uniform(-shear, shear)]) * np.pi / 180)
    S[1, 0] = torch.tan(torch.FloatTensor([random.uniform(-shear, shear)]) * np.pi / 180)

    T = torch.eye(3)
    T[0, 2] = torch.FloatTensor([random.uniform(0.5 - translate, 0.5 + translate)]) * width
    T[1, 2] = torch.FloatTensor([random.uniform(0.5 - translate, 0.5 + translate)]) * height

    M = T @ S @ R @ P @ C

    # Apply image transformation using OpenCV (numpy)
    if perspective:
        transformed_im = cv2.warpPerspective(im_np, M.numpy(), dsize=(width, height), borderValue=(114, 114, 114))
    else:
        transformed_im = cv2.warpAffine(im_np, M[:2].numpy(), dsize=(width, height), borderValue=(114, 114, 114))

    # Convert numpy array back to PIL image
    transformed_im_pil = Image.fromarray(transformed_im)


    targets = targets.copy()
    if "boxes" in targets:
        boxes = targets["boxes"].to(torch.float32)  # Ensure boxes are float for transformation
        boxes = box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh")
        boxes[:, :2] = torch.matmul(M[:2, :2], boxes[:, :2].T).T + M[:2, 2]
        boxes[:, 2:] *= s
        boxes = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        targets["boxes"] = boxes.clamp(min=0)

    if "conveyor_points" in targets:
        conveyor_points = targets["conveyor_points"].to(torch.float32)
        points_homog = torch.cat([conveyor_points[0], torch.ones(conveyor_points.shape[1], 1)], dim=1)
        transformed_points = torch.matmul(M, points_homog.T).T
        transformed_points = transformed_points[:, :2] / transformed_points[:, 2:3]
        targets["conveyor_points"] = transformed_points
    if "area" in targets:
        area = targets["area"]
        scaled_area = area * (s ** 2)  # Scale is squared because area is proportional to s^2
        targets["area"] = scaled_area
    h, w = im_np.shape[:2]
    targets["size"] = torch.tensor([h, w])

    return transformed_im_pil, targets


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target

class Perspective(object):
    # def __init__(self):

    def __call__(self, img, target):
        return random_perspective_pil(img, target)

class Augment_hsv(object):
    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5):
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, img, target):
        img = augment_hsv_pil(img,self.hgain, self.sgain, self.vgain)
        return img, target
class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)



class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        # 归一化conveyor_points
        if "conveyor_points" in target:
            conveyor_points = target["conveyor_points"]
            # 归一化操作，考虑到conveyor_points的形状为[1, 4, 2]
            conveyor_points = conveyor_points / torch.tensor([w, h], dtype=torch.float32)
            target["conveyor_points"] = conveyor_points
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
