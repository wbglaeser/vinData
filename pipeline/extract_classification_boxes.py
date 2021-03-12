# imports
import os
from pathlib import Path

import pandas as pd

def load_bounding_boxes(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df
