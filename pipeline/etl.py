from typing import List
from enum import Enum
from prefect import task, Flow, Parameter
import pandas as pd

from pipeline.extract_dicom_files import load_images, RawImage
from pipeline.extract_classification_boxes import load_bounding_boxes
from pipeline.image_processing import preprocess_images
from pipeline.load_to_tfRecords import load_to_tfRecords

from pipeline.pipeline_config import *

@task
def extract_image_data():
    images = load_images(path_to_images, Environment.local)
    return images

@task
def extract_classifcation_data():
    df = load_bounding_boxes(path_to_bounding_boxes)
    return df

@task
def preprocess_image_data(images: List[RawImage], df: pd.DataFrame):
    preprocessed_images = preprocess_images(images, df)
    return preprocessed_images

@task
def load_images_tfrecords(processed_images):
    load_to_tfRecords(processed_images, path_to_tfRecords)

# setup flow
with Flow("Dicom-ETL") as flow:
    environment = Parameter("environment", default= "local")
    raw_image_data = extract_image_data()
    df = extract_classifcation_data()
    processed_images = preprocess_image_data(raw_image_data, df)
    load_images_tfrecords(processed_images)

parameters = {
    "environment":"local"
}

flow.run()
#flow.register(project_name="vinData Challenge")
