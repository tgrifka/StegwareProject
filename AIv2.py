import math, re, os
import pathlib

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as L
import efficientnet.tfkeras as efn
from sklearn import metrics
from sklearn.model_selection import train_test_split


"""

    This is based on a notebook that was a posted solution to the Alaska2 competition which
    the dataset is from
    The notebook can be found here:https://www.kaggle.com/saurabhmaydeo/notebook50d155db1c

"""
if __name__ == '__main__':
    strategy = tf.distribute.get_strategy()
    AUTO = tf.data.experimental.AUTOTUNE

    # Configuration
    EPOCHS = 10
    BATCH_SIZE = 16 * strategy.num_replicas_in_sync

    # reads in dataset
    data_dir = 'D:/Stegware/startingSet'
    data_dir = pathlib.Path(data_dir)
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)

    batch_size = 32
    image_height = 512
    image_width = 512
    """
    def decode_image(filename, label=None, image_size=(512, 512)):
        bits = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(bits, channels=3)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(image, image_size)

        if label is None:
            return image
        else:
            return image, label


    def data_augment(image, label=None):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)

        if label is None:
            return image
        else:
            return image, label

    """
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(image_height, image_width),
        batch_size=batch_size)

    # creates the testing set

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(image_height, image_width),
        batch_size=batch_size)


    def build_lrfn(lr_start=0.00001, lr_max=0.000075,
                   lr_min=0.000001, lr_rampup_epochs=20,
                   lr_sustain_epochs=0, lr_exp_decay=.8):
        lr_max = lr_max * strategy.num_replicas_in_sync

        def lrfn(epoch):
            if epoch < lr_rampup_epochs:
                lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
            elif epoch < lr_rampup_epochs + lr_sustain_epochs:
                lr = lr_max
            else:
                lr = (lr_max - lr_min) * lr_exp_decay ** (epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min
            return lr

        return lrfn


    with strategy.scope():
        model = tf.keras.Sequential([
            efn.EfficientNetB3(
                input_shape=(512, 512, 3),
                weights='imagenet',
                include_top=False
            ),
            L.GlobalAveragePooling2D(),
            L.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        model.summary()

    #STEPS_PER_EPOCH = train_labels.shape[0] // BATCH_SIZE

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        #     callbacks=[lr_schedule],
        #     steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=val_ds
    )

    model.save("model.h5")


    def display_training_curves(training, validation, title, subplot):
        """
        Source: https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu
        """
        if subplot % 10 == 1:  # set up the subplots on the first call
            plt.subplots(figsize=(10, 10), facecolor='#F0F0F0')
            plt.tight_layout()
        ax = plt.subplot(subplot)
        ax.set_facecolor('#F8F8F8')
        ax.plot(training)
        ax.plot(validation)
        ax.set_title('model ' + title)
        ax.set_ylabel(title)
        # ax.set_ylim(0.28,1.05)
        ax.set_xlabel('epoch')
        ax.legend(['train', 'valid.'])


    display_training_curves(
        history.history['loss'],
        history.history['val_loss'],
        'loss', 211)
    display_training_curves(
        history.history['accuracy'],
        history.history['val_accuracy'],
        'accuracy', 212)