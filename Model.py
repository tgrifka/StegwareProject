import DCTCalc
import pathlib

import matplotlib.pyplot as plt
import tensorflow as tf
import os

from tensorflow.keras import layers
import efficientnet.tfkeras as efn
from tensorflow.keras.models import Sequential
"""
Things to be included in the models

Model 1&2: Just Coefficients of the 3 blocks

Model 3&4:

Coefficients of the 3 blcoks
Number of 0s between the blocks
Avg value of coefficients except DC value
Deviation from Avg Color - This may help or hurt unsure



"""

def main():
    i = DCTCalc.batch_calculate_dct(1)
    model1(i)
    # model2(i)

def model1(input):
    print("Running Model 1")
    batch_size = 8
    width = 512
    height = 512
    training_data_set = tf.data.Dataset.from_tensor_slices(input)
    print(training_data_set.shape)
    training_data_set.shuffle(123)

    model = Sequential([
        layers.Dense(50, activation='sigmoid', input_shape=(3, 512, 512)),
        layers.Dense(50, activation='sigmoid'),
        layers.Dense(50, activation='sigmoid'),
        layers.Dense(50, activation='sigmoid'),
        layers.Dense(50, activation='sigmoid'),
        layers.Dense(50, activation='sigmoid'),
        layers.Dense(50, activation='sigmoid'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])

    model.summary()

    epochs = 1
    history = model.fit(training_data_set, epochs=epochs)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def model2(input):
    print("Running Model 2")
    batch_size = 8
    width = 512
    height = 512
    training_data_set = tf.data.Dataset.from_tensor_slices(input)
    training_data_set.shuffle(123)

    model = Sequential([
        efn.EfficientNetB7(
            input_shape=(3, 512, 512),
            weights='imagenet',
            include_top=False
        ),
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])

    model.summary()

    epochs = 1
    history = model.fit(training_data_set, epochs=epochs)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


if __name__ == '__main__':
    main()
