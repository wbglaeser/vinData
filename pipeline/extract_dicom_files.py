# imports
import os
from typing import Dict, List
from pydantic import BaseModel
from pathlib import Path
import enum

from pydicom import dcmread
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

from pipeline.pipeline_config import *
from settings import *

class RawImage(BaseModel):
    id: str
    pixel_array: np.ndarray

    class Config:
        arbitrary_types_allowed = True

def load_raw_image(path: Path, environment: Environment) -> RawImage:

    if environment == Environment.local:
        ds = dcmread(path)
        image = ds.pixel_array

    elif environment == Environment.google_cloud:
        image_bytes = tf.io.read_file(path)
        image = tfio.image.decode_dicom_image(image_bytes, dtype=tf.uint16, on_error="skip").numpy()

    # parse into RawImageModel
    new_image = {
        "id": path.split("/")[-1].replace(".dicom", ""),
        "pixel_array": image
    }
    new_image_typed = RawImage(**new_image)

    return new_image_typed

def load_images(path: Path, environment: Environment) -> List[RawImage]:

    container = []
    for fname in os.listdir(path):
        if ".dicom" in fname:
            fpath = os.path.join(path, fname)
            new_image = load_raw_image(fpath, environment)
            container.append(new_image)

    return container
