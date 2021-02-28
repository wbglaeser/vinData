import os

import tensorflow as tf
import numpy as np

from pipeline.load_to_tfRecords import LoadToTFRecords
from trainer.model.encode_labels import LabelEncoder
from trainer.model.utils import get_backbone
from trainer.model.retina_net import RetinaNet
from trainer.model.retina_loss import RetinaNetLoss
from settings import *

label_encoder = LabelEncoder()
model_dir = "retinanet/"


# Filename
FILENAME = "training.tfrecords"
RAW_TRAIN_DIR = "train"
RAW_TRAIN_DIR_FULL = os.path.join(DATA_DIR, RAW_TRAIN_DIR)
FILEPATH = os.path.join(RAW_TRAIN_DIR_FULL, FILENAME)

# Load dataset
autotune = tf.data.experimental.AUTOTUNE
train_dataset = tf.data.TFRecordDataset(FILEPATH)
parsed_train_dataset = train_dataset.map(LoadToTFRecords.read_and_decode)
print("NOW TEST TEST TEST")
for (image_px, boxes, labels) in parsed_train_dataset:
    print(image_px)
    print(boxes)
    print(labels)


parsed_train_dataset = parsed_train_dataset.map(
    label_encoder.encode_batch, num_parallel_calls=autotune
)
parsed_train_dataset = parsed_train_dataset.apply(tf.data.experimental.ignore_errors())
parsed_train_dataset = parsed_train_dataset.prefetch(autotune)

val_dataset = tf.data.TFRecordDataset(FILEPATH)
autotune = tf.data.experimental.AUTOTUNE
val_dataset = val_dataset.map(
    label_encoder.encode_batch, num_parallel_calls=autotune
)
val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
val_dataset = val_dataset.prefetch(autotune)

# load model
num_classes = 14
batch_size = 2
epochs = 1

# define callback
callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
        monitor="loss",
        save_best_only=False,
        save_weights_only=True,
        verbose=1,
    )
]

# schedule learning rate
learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
learning_rate_boundaries = [125, 250, 500, 240000, 360000]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)

# build model
resnet50_backbone = get_backbone()
loss_fn = RetinaNetLoss(num_classes)
model = RetinaNet(num_classes, resnet50_backbone)

optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
model.compile(loss=loss_fn, optimizer=optimizer)

# fit model
model.fit(
    train_dataset.take(100),
    validation_data=val_dataset.take(50),
    epochs=epochs,
    callbacks=callbacks_list,
    verbose=1,
)
