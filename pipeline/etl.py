from typing import List

from prefect import task, Flow, Parameter
import pandas as pd

from pipeline.extract_dicom_files import ExtractDicomImages, RawImage
from pipeline.extract_classification_boxes import ExtractClassificationBoxes
from pipeline.preprocess_image_data import PreprocessImageData
from pipeline.load_to_tfRecords import LoadToTFRecords


@task
def extract_image_data(environment):

    extract_images = ExtractDicomImages(environment)
    images = extract_images.run()
    return images

@task
def extract_classifcation_data(environment):
    extract_boxes = ExtractClassificationBoxes(environment)
    df = extract_boxes.run()
    return df

@task
def preprocess_image_data(images: List[RawImage], df: pd.DataFrame):
    preprocesser = PreprocessImageData()
    preprocess_images = preprocesser.run(images, df)
    return preprocess_images

@task
def load_to_tfRecords(processed_images):
    loader = LoadToTFRecords()
    loader.run(processed_images)

# setup flow
with Flow("Dicom-ETL") as flow:
    environment = Parameter("environment", default= "local")
    raw_image_data = extract_image_data(environment)
    df = extract_classifcation_data(environment)
    processed_images = preprocess_image_data(raw_image_data, df)
    load_to_tfRecords(processed_images)

parameters = {
    "environment":"local"
}

flow.run(parameters=parameters)
#flow.register(project_name="vinData Challenge")
