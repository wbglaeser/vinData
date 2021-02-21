#!/usr/bin/env python3

"""Model to classify draft beers

This file contains all the model information: the training steps, the batch
size and the model itself.
"""

import tensorflow as tf
import tensorflow_hub as hub

feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

def get_batch_size():
    """Returns the batch size that will be used by your solution.
    It is recommended to change this value.
    """
    return 64

def get_epochs():
    """Returns number of epochs that will be used by your solution.
    It is recommended to change this value.
    """
    return 150

def get_final_epochs():
    """Returns number of epochs that will be used by your solution.
    It is recommended to change this value.
    """
    return 90

def solution(input_layer):
    """Returns a compiled model.

    This function is expected to return a model to identity the different beers.
    The model's outputs are expected to be probabilities for the classes and
    and it should be ready for training.
    The input layer specifies the shape of the images. The preprocessing
    applied to the images is specified in data.py.

    Add your solution below.

    Parameters:
        input_layer: A tf.keras.layers.InputLayer() specifying the shape of the input.
            RGB colored images, shape: (width, height, 3)
    Returns:
        model: A compiled model
    """
    
    # TODO: Code of your solution
    num_classes = 5
    IMG_SIZE = 224
    img_shape = (IMG_SIZE, IMG_SIZE) + (3,)

    data_diversification = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
    ])

    resize_and_rescale = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    ])

    feature_extractor_layer = hub.KerasLayer(
        feature_extractor_model, input_shape=img_shape, trainable=False
    )

    classification_layer = tf.keras.Sequential([
        tf.keras.layers.Dropout(.7),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # build model
    #x = data_diversification(input_layer) 
    x = resize_and_rescale(input_layer)
    x = feature_extractor_layer(x)
    outputs = classification_layer(x)
    model = tf.keras.Model(input_layer, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['acc']
    )

    # TODO: Return the compiled model
    return model
