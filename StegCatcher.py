import tkinter as tk
from tkinter.filedialog import askdirectory, askopenfilenames
import pathlib

import matplotlib.pyplot as plt
import tensorflow as tf
import os

from keras.models import model_from_json
from tensorflow.keras import layers
import efficientnet.tfkeras as efn
from tensorflow.keras.models import Sequential


def main():

    image_height = 512
    image_width = 512

    def load_model():

        global model

        json_file = open('model.json', 'r')
        model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights("model.h5")
        model._make_predict_function()

    def select_path(do_dir):
        files = [('jpg, *.jpg')]
        if do_dir:
            file = askdirectory()
        else:
            file = askopenfilenames(filetypes=files, defaultextension=files)
        input_e.insert(0, file)



    window = tk.Tk()
    window.title("StegCatcher")

    window.columnconfigure(0, weight=0)
    window.columnconfigure(1, weight=1)
    window.columnconfigure(2, weight=0)
    window.rowconfigure(0, weight=1)

    window.rowconfigure(1, weight=0)
    window.config(background='#688387')

    input_l = tk.Label(window, text='Folder/File Path', background='#688387', foreground='#000000')
    input_l.grid(column=0, row=0)

    input_e = tk.Entry(window)
    input_e.grid(column=1, row=0)
    do_dir = tk.IntVar()
    button_frame = tk.Frame(background='#688387', relief='raised', height=4)
    button_frame.grid(column=0, row=1, sticky='s', columnspan=3)
    folder_check = tk.Checkbutton(button_frame, text='Folder?', variable=do_dir)
    folder_check.grid(column=0, row=0, sticky='s', padx=5, pady=5)

    button_select = tk.Button(window, text='Select File/Directory', command=lambda: select_path(do_dir.get()),
                              background='#0ba6a3', foreground='#142c2b')
    button_select.grid(column=2, row=0)

    """
    settings_btn = tk.Button(button_frame, text='Settings', background='#0ba6a3', foreground='#142c2b', width=12)
    settings_btn.grid(column=0, row=0, sticky='s', padx=5, pady=5)
    """

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
    checkpoint_path = 'D:\Stegware\StegwareProject\\training_checkpoints\\combind.data-00000-of-00001'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    model.load_weights(checkpoint_path)

    def run(path):
        if os.path.isdir(path):
            data_dir = pathlib.Path('D:\Stegware\FeatureTesting')
            ds = tf.keras.preprocessing.image_dataset_from_directory(
                data_dir,
                validation_split=0.2,
                subset="training",
                seed=123,
                image_size=(image_height, image_width),
                batch_size=4)
        res = model.predict(ds)
        print(res)

    run_btn = tk.Button(button_frame, text='Run', background='#0ba6a3', foreground='#142c2b', width='12',
                        command=lambda: run(input_e.get()))
    run_btn.grid(column=1, row=0, sticky='s', padx=5, pady=5)

    window.mainloop()


if __name__ == '__main__':
    main()