from typing import List

from prefect import task, Flow
import pandas as pd

from pipeline.extract_dicom_files import ExtractDicomImages, RawImage
from pipeline.extract_classification_boxes import ExtractClassificationBoxes
from pipeline.preprocess_image_data import PreprocessImageData
from pipeline.load_to_tfRecords import LoadToTFRecords

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
    preprocess_images = PreprocessImageData.run(images, df)
    return preprocess_images

@task
def load_to_tfRecords(processed_images):
    LoadToTFRecords.run(processed_images)

# setup flow
with Flow("Dicom-ETL") as flow:
    raw_image_data = extract_image_data()
    df = extract_classifcation_data()
    processed_images = preprocess_image_data(raw_image_data, df)
    load_to_tfRecords(processed_images)

#flow.run()
flow.register(project_name="vinData Challenge")
