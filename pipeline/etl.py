from typing import List

from prefect import task, Flow, Parameter
import pandas as pd

from pipeline.extract_dicom_files import ExtractDicomImages, RawImage
from pipeline.extract_classification_boxes import ExtractClassificationBoxes
from pipeline.preprocess_image_data import PreprocessImageData
from pipeline.load_to_tfRecords import LoadToTFRecords

@task
def extract_image_data(environment):
    images = ExtractDicomImages.run(environment)
    return images

@task
def extract_classifcation_data(environment):
    df = ExtractClassificationBoxes.run(environment)
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
    environment = Parameter("environment", default= "local")
    raw_image_data = extract_image_data(environment)
    df = extract_classifcation_data(environment)
    processed_images = preprocess_image_data(raw_image_data, df)
    load_to_tfRecords(processed_images)

parameters = {"environment":"local"}
flow.run(parameters=parameters)
#flow.register(project_name="vinData Challenge")
