import cv2
import math
import numpy as np
from PIL import Image


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


angle = 180
img = Image.open('path/to/your/image.jpg').convert('RGB')

instances = [
    {'category_id': 0, 'bbox': [693, 531, 90, 73], 'image_id': 20211624},
    {'category_id': 2, 'bbox': [778, 608, 75, 261], 'image_id': 20211624}
]

conveyor_points = [(100, 100), (200, 100), (200, 200), (100, 200)]

rotated_img, rotated_instances, rotated_conveyor_points = rotate(angle, img, instances, conveyor_points)
print(rotated_instances)
print(rotated_conveyor_points)
