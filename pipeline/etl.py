from typing import List

from prefect import task, Flow
import pandas as pd

from pipeline.extract_dicom_files import ExtractDicomImages, RawImage
from pipeline.extract_classification_boxes import ExtractClassificationBoxes
from pipeline.preprocess_image_data import PreprocessImageData

@task
def extract_image_data():
    images = ExtractDicomImages.run()
    return images

@task
def extract_classifcation_data():
    df = ExtractClassificationBoxes.run()
    return df

@task
def preprocess_image_data(images: List[RawImage], df: pd.DataFrame):
    PreprocessImageData.run(images, df)

# setup flow
with Flow("Dicom-ETL") as flow:
    raw_image_data = extract_image_data()
    df = extract_classifcation_data()
    preprocess_image_data(raw_image_data, df)

flow.run()
#flow.register(project_name="vinData Challenge")
