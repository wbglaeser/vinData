# imports
import os
from typing import List
from pydantic import BaseModel

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as pac

from pipeline.extract_dicom_files import RawImage

class ObjectSpecs(BaseModel):
    labels: np.ndarray
    bboxes: np.ndarray

    class Config:
        arbitrary_types_allowed = True

class FeedImage(BaseModel):
    pixel_array: np.ndarray
    bits_stored: int
    objects: ObjectSpecs

    class Config:
        arbitrary_types_allowed = True

class PreprocessImageData():

    @classmethod
    def run(cls, raw_images: List[RawImage], df: pd.DataFrame):

        # instantiate class
        ppi = cls()

        # run through all images
        ppi.preprocess_all_images(raw_images, df)

    def add_boxes_to_image(self, image: RawImage, df: pd.DataFrame) -> FeedImage:

        # retrieve unique image identifier
        uid = image.id

        # set pixel array to appropriate dimensions
        pixel_array = np.expand_dims(image.pixel_array, axis=2)

        # extract class labels
        labels = df[df["image_id"] == uid]["class_id"].values

        # extract bounding boxes
        bboxes = df[df["image_id"] == uid][["x_min", "y_min", "x_max", "y_max"]].values[:2].astype(np.float32)

        # parse to new image object
        feed_image = {
            "pixel_array": pixel_array,
            "objects": {
                "labels": labels,
                "bboxes": bboxes
            },
            "bits_stored": image.bits_stored
        }
        new_feed_image = FeedImage(**feed_image)

        return new_feed_image

    def preprocess_all_images(self, images: List[RawImage], df: pd.DataFrame):

        for image in images:

            # preprocess this shit
            image_with_boxes = self.add_boxes_to_image(image, df)
            image, bboxes = self.preprocess_image(image_with_boxes)

    def preprocess_image(self, image: FeedImage):

        # split into unique vars for processing
        image_px = image.pixel_array
        bboxes = image.objects.bboxes
        class_id = tf.cast(image.objects.labels, dtype=tf.int32)

        self.plot_image_with_box(image_px, bboxes)

        # normalise pixel value
        image_px = image_px / image.bits_stored

        print("beforeshape: ", image_px.shape)
        print("beforeboxes: ", bboxes)

        # random flip horizontal
        image_px, bboxes = self.random_flip_horizontal(image_px, bboxes)

        print("intermediateshape: ", image_px.shape)
        print("intermediateboxes: ", bboxes)

        # reshape
        image_px, bboxes = self.resize_and_pad_image(image_px, bboxes)

        print("aftershape: ", image_px.shape)
        print("afterboxes: ", bboxes)

        self.plot_image_with_box(image_px, bboxes)

        return image_px, bboxes

    def random_flip_horizontal(self, image, boxes):

        # 50% chance we flip the image
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            boxes = tf.stack(
                [image.shape[1] - boxes[:, 0], boxes[:, 1], image.shape[1] - boxes[:, 2], boxes[:, 3]], axis=-1
            )
        return image, boxes

    def resize_and_pad_image(self, image, bboxes, min_side=800.0, max_side=1333.0, stride=128.0):

        # retrieve current shape
        image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)

        # compute ratio for min side
        ratio = min_side / tf.reduce_min(image_shape)

        # check if this solves shape for max side as well
        if ratio * tf.reduce_max(image_shape) > max_side:
            ratio = max_side / tf.reduce_max(image_shape)

        # new image shape
        image_shape = ratio * image_shape
        image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))

        # apply reshape to boxes
        bboxes = tf.stack(
            [
                bboxes[:, 0] * ratio,
                bboxes[:, 1] * ratio,
                bboxes[:, 2] * ratio,
                bboxes[:, 3] * ratio,
            ],
            axis=-1,
        )

        # pad image to comply with stride
        padded_image_shape = tf.cast(
            tf.math.ceil(image_shape / stride) * stride, dtype=tf.int32
        )
        image = tf.image.pad_to_bounding_box(
            image, 0, 0, padded_image_shape[0], padded_image_shape[1]
        )

        return image, bboxes

    def plot_image_with_box(self, image_px, bboxes):
        plt.imshow(image_px, cmap=plt.cm.gray)

        for box in bboxes:
            height = box[3] - box[1]
            width = box[2] - box[0]
            rectangle = pac.Rectangle((box[0],box[1]), width, height,ec="red", fill=False)
            plt.gca().add_patch(rectangle)

        plt.show()

def swap_xy(boxes):
    """Swaps order the of x and y coordinates of the boxes.

    Arguments:
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes.

    Returns:
      swapped boxes with shape same as that of boxes.
    """
    return tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)


def convert_to_xywh(boxes):
    """Changes the box format to center, width and height.

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[xmin, ymin, xmax, ymax]`.

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
        axis=-1,
    )

def convert_to_corners(boxes):
    """Changes the box format to corner coordinates

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[x, y, width, height]`.

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,
    )