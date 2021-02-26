# imports
import os
from pydantic import BaseModel

import pandas as pd

from settings import *

class ExtractClassificationBoxes():

    FILE_NAME = "train.csv"

    def load_data_frame(self) -> pd.DataFrame:

        # load classes with bounding boxes from csv
        fpath = os.path.join(DATA_DIR, self.FILE_NAME)
        df = pd.read_csv(fpath)

        return df

    @classmethod
    def run(cls) -> pd.DataFrame:

        # instantiate class
        ecb = cls()

        # loop through images
        df = ecb.load_data_frame()

        return df
