#!/usr/bin/env python3

"""Train and evaluate the model

This file trains the model upon the training data and evaluates it with
the eval data.
It uses the arguments it got via the gcloud command.
"""

import os
import argparse
import logging
from datetime import datetime

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import trainer.data as data
import trainer.model as model

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

def train_model(params):
    """The function gets the training data from the training folder,
    the evaluation data from the eval folder and trains your solution
    from the model.py file with it.

    Parameters:
        params: parameters for training the model
    """
    (train_data, train_labels) = data.create_data_with_labels("data/train/")
    (eval_data, eval_labels) = data.create_data_with_labels("data/eval/")

    img_shape = train_data.shape[1:]
    input_layer = tf.keras.Input(shape=img_shape, name='input_image')

    ml_model = model.solution(input_layer)
    
    if ml_model is None:
        print("No model found. You need to implement one in model.py")
    else:
        ml_model.fit(
            train_data, train_labels,
            validation_data=(eval_data, eval_labels),
            batch_size=model.get_batch_size(),
            epochs=model.get_epochs(),
            callbacks=[
                tensorboard_callback,
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            ]
        )
        ml_model.evaluate(eval_data, eval_labels, verbose=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    tf_logger = logging.getLogger("tensorflow")
    tf_logger.setLevel(logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf_logger.level // 10)

    train_model(args)
