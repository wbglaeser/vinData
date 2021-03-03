# imports
import os
from pydantic import BaseModel

import pandas as pd

from settings import *

class ExtractClassificationBoxes():

    FILE_NAME = "train.csv"

    def __init__(self, environment: str):
        self.environment = environment

    def load_data_frame(self) -> pd.DataFrame:

        # load classes with bounding boxes from csv
        if self.environment == "local":
            fpath = os.path.join(DATA_DIR, self.FILE_NAME)
    
        elif self.environment == "gcp":
            fpath = os.path.join(DATA_DIR_GCP, self.FILE_NAME)
            
        df = pd.read_csv(fpath)

        return df

    @classmethod
    def run(cls, environment) -> pd.DataFrame:

        # instantiate class
        ecb = cls(environment)

        # loop through images
        df = ecb.load_data_frame()

        return df
