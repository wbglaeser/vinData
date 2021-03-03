# imports
import os
from typing import Dict, List
from pydantic import BaseModel

from pydicom import dcmread
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

from settings import *

class RawImage(BaseModel):
    id: str
    pixel_array: np.ndarray

    class Config:
        arbitrary_types_allowed = True

class ExtractDicomImages():
    
    RAW_TRAIN_DIR = "train"
    RAW_TRAIN_DIR_FULL = os.path.join(DATA_DIR, RAW_TRAIN_DIR)
    RAW_TRAIN_DIR_FULL_GCP = os.path.join(DATA_DIR_GCP, RAW_TRAIN_DIR)

    def __init__(self, environment: str):
        self.environment = environment

    def load_image(self, path: str) -> RawImage:

        if self.environment == "local":
            ds = dcmread(path)
            image = ds.pixel_array

        elif self.environment == "gcp":
            image_bytes = tf.io.read_file(path)
            image = tfio.image.decode_dicom_image(image_bytes, dtype=tf.uint16, on_error="skip").numpy()
        
        # parse into RawImageModel
        new_image = {
            "id": path.split("/")[-1].replace(".dicom", ""),
            "pixel_array": image
        }
        new_image_typed = RawImage(**new_image)

        return new_image_typed

    def load_all_images(self) -> List[RawImage]:

        container = []
        fpath = None
        if self.environment == "local":
            for fname in os.listdir(self.RAW_TRAIN_DIR_FULL):
                if ".dicom" in fname:
                    fpath = os.path.join(self.RAW_TRAIN_DIR_FULL, fname)
                    new_image = self.load_image(fpath)
                    container.append(new_image)
        
        elif self.environment == "gcp":
            for fname in tf.io.gfile.listdir(self.RAW_TRAIN_DIR_FULL_GCP):
                if ".dicom" in fname:
                    fpath = os.path.join(self.RAW_TRAIN_DIR_FULL_GCP, fname)
                    new_image = self.load_image(fpath)
                    container.append(new_image)
        
        return container

    @classmethod
    def run(cls, environment:str) -> List[RawImage]:

        # instantiate class
        edi = cls(environment)

        # loop through images
        images = edi.load_all_images()

        return images

if __name__ == "__main__":
    ExtractDicomImages.run()
