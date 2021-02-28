# imports
import os
from typing import List
from pydantic import BaseModel

import pandas as pd
import numpy as np
import tensorflow as tf

from pipeline.preprocess_image_data import ProcessedContainer
from settings import *


class LoadToTFRecords():

    RAW_TRAIN_DIR = "train"
    RAW_TRAIN_DIR_FULL = os.path.join(DATA_DIR, RAW_TRAIN_DIR)

    @classmethod
    def run(cls, container: List[ProcessedContainer]):

        lttfr = cls()
        lttfr.load_to_tfRecords(container)

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _int64_list_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    # images and labels array as input
    def load_to_tfRecords(self, images: List[ProcessedContainer]):

        filename = os.path.join(self.RAW_TRAIN_DIR_FULL, 'training.tfrecords')

        with tf.io.TFRecordWriter(filename) as file_writer:
            for image in images:
                image_raw = image.image_px.numpy().tostring()
                bboxes_raw = image.bboxes.numpy().tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': self._int64_feature(image.image_px.shape[0]),
                    'width': self._int64_feature(image.image_px.shape[1]),
                    'depth': self._int64_feature(1),
                    'labels': self._int64_list_feature(image.labels),
                    'boxes_raw': self._bytes_feature(bboxes_raw),
                    'image_raw': self._bytes_feature(image_raw)}))
                file_writer.write(example.SerializeToString())
