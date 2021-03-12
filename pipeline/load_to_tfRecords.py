# imports
import os
from typing import List
from pydantic import BaseModel
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf

from settings import *

def _int64_list_feature(alue):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# images and labels array as input
def load_to_tfRecords(images: List[dict], path: Path):

    with tf.io.TFRecordWriter(path) as file_writer:
        for image in images:
            labels_raw = image["labels"].numpy().tostring()
            image_raw = image["image_px"].numpy().tostring()
            bboxes_raw = image["boxes"].numpy().tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(image["image_px"].shape[0]),
                'width': _int64_feature(image["image_px"].shape[1]),
                'depth': _int64_feature(1),
                'labels_length': _int64_feature(image["labels"].shape[0]),
                'labels_raw': _bytes_feature(labels_raw),
                'boxes_raw': _bytes_feature(bboxes_raw),
                'image_raw': _bytes_feature(image_raw)}))
            file_writer.write(example.SerializeToString())

def read_and_decode(example_proto):

    # define features for re-parsing
    features={
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'labels_length': tf.io.FixedLenFeature([], tf.int64),
        'labels_raw': tf.io.FixedLenFeature([], tf.string),
        'boxes_raw': tf.io.FixedLenFeature([], tf.string),
        'image_raw': tf.io.FixedLenFeature([], tf.string)
    }
    features = tf.io.parse_single_example(
        example_proto, features
    )

    # decode image
    image_px = tf.io.decode_raw(features['image_raw'], tf.float32)
    image_px = tf.reshape(image_px, (features['height'], features['width'], features['depth']))

    # decode boxes
    boxes = tf.io.decode_raw(features['boxes_raw'], tf.float32)
    boxes = tf.reshape(boxes, (features['labels_length'], 4))

    # decode labels
    labels = tf.io.decode_raw(features['labels_raw'], tf.int32)

    return image_px, boxes, labels
