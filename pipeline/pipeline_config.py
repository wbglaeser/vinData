from enum import Enum

class Environment(Enum):
    local = "local"
    google_cloud = "google_cloud"

path_to_images = "data/train"
path_to_bounding_boxes = "data/train.csv"
