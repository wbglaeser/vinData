# imports
import os
from typing import Dict, List
from pydantic import BaseModel

from pydicom import dcmread
import numpy as np

from settings import *

class RawImage(BaseModel):
    id: str
    bits_stored: int
    pixel_array: np.ndarray

    class Config:
        arbitrary_types_allowed = True

class ExtractDicomImages():

    RAW_TRAIN_DIR = "train"
    RAW_TRAIN_DIR_FULL = os.path.join(DATA_DIR, RAW_TRAIN_DIR)

    def load_image(self, path: str) -> RawImage:

        # load image
        ds = dcmread(path)

        # parse into RawImageModel
        new_image = {
            "id": path.split("/")[-1].replace(".dicom", ""),
            "bits_stored": ds.BitsStored,
            "pixel_array": ds.pixel_array
        }
        new_image_typed = RawImage(**new_image)

        return new_image_typed

    def load_all_images(self) -> List[RawImage]:

        container = []

        # loop through directory
        for fname in os.listdir(self.RAW_TRAIN_DIR_FULL):
            if ".dicom" in fname:
                fpath = os.path.join(self.RAW_TRAIN_DIR_FULL, fname)

                # parse image to raw image format
                new_image = self.load_image(fpath)
                container.append(new_image)

        return container

    @classmethod
    def run(cls) -> List[RawImage]:

        # instantiate class
        edi = cls()

        # loop through images
        images = edi.load_all_images()

        return images

if __name__ == "__main__":
    ExtractDicomImages.run()
