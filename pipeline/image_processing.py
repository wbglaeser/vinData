# imports
import os
from typing import List
from pydantic import BaseModel

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
import matplotlib.pyplot as plt
import matplotlib.patches as pac

from pipeline.extract_dicom_files import RawImage

def preprocess_images(images: List[RawImage], df: pd.DataFrame) -> List:

    container = []
    for image in images:

        # retrieve image pixels
        image_px = adjust_image_dtype(image)

        # retrieve bounding boxes with labels
        labels, boxes = match_bounding_boxes(image, df)

        # skip if no finding
        if 14 in labels: continue

        # else continue processing
        processed_container = preprocess_image(image_px, boxes, labels)
        container.append(processed_container)

    return container

def adjust_image_dtype(image: RawImage):
    return np.expand_dims(image.pixel_array, axis=2)

def match_bounding_boxes(image: RawImage, df: pd.DataFrame):

    # extract class labels
    labels = df[df["image_id"] == image.id]["class_id"].values
    labels = tf.cast(labels, dtype=tf.int32)

    # extract bounding boxes
    bboxes = df[df["image_id"] == image.id][["x_min", "y_min", "x_max", "y_max"]].values.astype(np.float32)

    return labels, bboxes

def preprocess_image(image_px: np.ndarray, boxes: np.ndarray, labels: List):

    plot_image_with_box(image_px, boxes)

    # random flip horizontal
    image_px, boxes = random_flip_horizontal(image_px, boxes)

    # reshape
    image_px, boxes = resize_and_pad_image(image_px, boxes)

    plot_image_with_box(image_px, boxes)

    # convert boxes to xywh format
    boxes = convert_to_xywh(boxes)

    # wrap in image type
    image = {
        "image_px": image_px,
        "labels": labels,
        "bboxes": boxes
    }

    return image

def random_flip_horizontal(image_px: np.ndarray, boxes: np.ndarray):

    # 50% chance we flip the image
    if tf.random.uniform(()) > 0.5:
        image_px = tf.image.flip_left_right(image_px)
        boxes = tf.stack(
            [
                image_px.shape[1] - boxes[:, 0],
                boxes[:, 1],
                image_px.shape[1] - boxes[:, 2],
                boxes[:, 3]], axis=-1
        )

    return image_px, boxes

def resize_and_pad_image(
    image_px: np.ndarray,
    boxes: np.ndarray,
    min_side=800.0,
    max_side=1333.0,
    stride=128.0):

    # retrieve current shape
    image_shape = tf.cast(tf.shape(image_px)[:2], dtype=tf.float32)

    # compute ratio for min side
    ratio = min_side / tf.reduce_min(image_shape)

    # check if this solves shape for max side as well
    if ratio * tf.reduce_max(image_shape) > max_side:
        ratio = max_side / tf.reduce_max(image_shape)

    # new image shape
    image_shape = ratio * image_shape
    image_px = tf.image.resize(image_px, tf.cast(image_shape, dtype=tf.int32))

    print(ratio)
    print(boxes)
    # apply reshape to boxes
    boxes = tf.stack(
        [
            boxes[:, 0] * ratio,
            boxes[:, 1] * ratio,
            boxes[:, 2] * ratio,
            boxes[:, 3] * ratio,
        ],
        axis=-1,
    )

    # pad image to comply with stride
    padded_image_shape = tf.cast(
        tf.math.ceil(image_shape / stride) * stride, dtype=tf.int32
    )
    image_px = tf.image.pad_to_bounding_box(
        image_px, 0, 0, padded_image_shape[0], padded_image_shape[1]
    )

    return image_px, boxes

def plot_image_with_box(image_px: np.ndarray, boxes: np.ndarray):
    plt.imshow(image_px, cmap=plt.cm.gray)

    for box in boxes:
        height = box[3] - box[1]
        width = box[2] - box[0]
        rectangle = pac.Rectangle((box[0],box[1]), width, height,ec="red", fill=False)
        plt.gca().add_patch(rectangle)

    plt.show()

def convert_to_xywh(boxes: np.ndarray):
    """Changes the box format to center, width and height."""

    x = (boxes[...,0] + boxes[..., 2]) / 2
    y = (boxes[...,1] + boxes[..., 3]) / 2
    w = boxes[...,3] - boxes[..., 1]
    h = boxes[...,3] - boxes[..., 1]

    return tf.stack([x, y, w, h], axis=-1)

def convert_to_corners(boxes: np.ndarray):
    """Changes the box format to corner coordinates"""
    t = tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,
    )
    return t

def convert_to_rgb(image, boxes, class_ids):
    "convert our grayscale images to rgb values"
    image_rgb = tf.image.grayscale_to_rgb(image)
    return image_rgb, boxes, class_ids
