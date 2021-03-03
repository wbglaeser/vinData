import os

import tensorflow as tf
import numpy as np

from pipeline.load_to_tfRecords import LoadToTFRecords
from pipeline.preprocess_image_data import PreprocessImageData
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
batch_size=2
autotune = tf.data.experimental.AUTOTUNE

# process data
dataset = tf.data.TFRecordDataset(FILEPATH)
dataset_size = sum(1 for _ in dataset)

dataset = dataset.map(LoadToTFRecords.read_and_decode)
dataset = dataset.shuffle(8 * batch_size)
dataset = dataset.padded_batch(
    batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True
)

c = 0
for _, boxes, class_ids in dataset:
    if c > 2: break
    print(boxes.shape)
    c += 1

dataset = dataset.map(PreprocessImageData.convert_to_rgb)
dataset = dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)

c = 0
for _, boxes in dataset:
    if c > 2: break
    print(boxes.shape)
    c += 1

dataset = dataset.apply(tf.data.experimental.ignore_errors())
dataset = dataset.prefetch(autotune)



# create train test split
train_size = int(0.8 * dataset_size)
val_size = int(0.2 * dataset_size)

#print(sum(1 for _ in dataset))

dataset = dataset.shuffle(8 * batch_size)
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)
val_dataset = val_dataset.take(val_size)

##print(sum(1 for _ in train_dataset))
#print(sum(1 for _ in val_dataset))

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
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=callbacks_list,
    verbose=1,
)
