#!/usr/bin/env python

import tensorflow as tf
import tensorflow_datasets as tfds
from functools import cache
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from os.path import exists
import numpy as np

IMAGE_SIZE = (128, 128)


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


def load_image(datapoint):
    input_image = tf.image.resize(datapoint['image'], IMAGE_SIZE)
    input_mask = tf.image.resize(datapoint['segmentation_mask'], IMAGE_SIZE)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


@cache
def load_dataset():
    dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
    train = dataset['train'].map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    test = dataset['test'].map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return (train, test), info


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3, ))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2,
                                 padding="same")(previous_block_activation)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # [Second half of the network: upsampling inputs] #

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes,
                            3,
                            activation="softmax",
                            padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


def load_model(
    *,
    force_training=False,
    epochs=3,
    continue_training=True,
    model_path="model",
):
    if not force_training and exists(model_path):
        return keras.models.load_model(model_path)

    if continue_training and exists(model_path):
        model = keras.models.load_model(model_path)
    else:
        model = get_model(IMAGE_SIZE, 3)
        model.compile(optimizer="rmsprop",
                      loss="sparse_categorical_crossentropy",
                      metrics=['sparse_categorical_accuracy'])

    (train_images, test_images), info = load_dataset()
    TRAIN_LENGTH = info.splits['train'].num_examples
    BATCH_SIZE = 64
    BUFFER_SIZE = 1000
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

    train_batches = train_images \
        .cache().shuffle(BUFFER_SIZE) \
        .batch(BATCH_SIZE).repeat() \
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    test_batches = test_images.batch(BATCH_SIZE)

    model.fit(
        train_batches,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=epochs,
        validation_data=test_batches,
    )
    model.save(model_path)

    return model


def create_mask(pred_mask):
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask


def test_model(model):
    (train_images, test_images), _ = load_dataset()
    for sample_image, sample_mask in test_images.shuffle(1000).batch(64).take(
            1):
        sample_pred = model.predict(sample_image)
        for image, mask, pred in zip(sample_image, sample_mask, sample_pred):
            display([
                image,
                mask,
                create_mask(pred),
            ])


def main():
    model = load_model()
    test_model(model)


if __name__ == '__main__':
    main()
