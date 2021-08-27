import pathlib

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers
import efficientnet.tfkeras as efn
from tensorflow.keras.models import Sequential


"""

    This is based on a notebook that was a posted solution to the Alaska2 competition which
    the dataset is from
    The notebook can be found here:https://www.kaggle.com/saurabhmaydeo/notebook50d155db1c

"""

if __name__ == '__main__':

    # reads in dataset
    data_dir = 'D:\Stegware\FeatureTesting'  # Update in lab
    data_dir = pathlib.Path(data_dir)
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)

    batch_size = 8
    image_height = 512
    image_width = 512

    # creates the training set

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

    # prints class names - not needed but a nice feature

    class_names = train_ds.class_names
    print(class_names)

    # prints out something... I am not sure what exactly

    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    # brings the dataset into memory for faster performance

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # creates a saving path for the NN weights

    checkpoint_path = 'D:\Stegware\StegwareProject\\training_checkpoints\\1'
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    # creates the model and normalizes the pixel information from 0-255 to 0-1

    num_classes = 2

    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(image_height, image_width, 3)),
        efn.EfficientNetB7(
            input_shape=(512, 512, 3),
            weights='imagenet',
            include_top=False
        ),
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation='sigmoid')
    ])

    # compiles the model

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # views all layers of the model

    model.summary()

    # trains the model with a set number of epochs

    epochs = 5
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[cp_callback])

    # displays the Training Accuracy

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()