from enum import Enum

class Environment(Enum):
    local = "local"
    google_cloud = "google_cloud"

# Define file paths
path_to_images = "data/train"
path_to_bounding_boxes = "data/train.csv"
path_to_tfRecords = "data/train.tfRecords"
