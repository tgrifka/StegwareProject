import glob
import tkinter as tk
from datetime import datetime
from tkinter.filedialog import askdirectory, askopenfilenames
import pathlib
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from keras.models import model_from_json
from tensorflow.keras import layers
import efficientnet.tfkeras as efn
from tensorflow.keras.models import Sequential

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512


def main():
    global model

    def load_model():
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
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
        efn.EfficientNetB7(
            input_shape=(512, 512, 3),
            weights='imagenet',
            include_top=False
        ),
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation='sigmoid')
    ])
    checkpoint_path = 'D:\\Stegware\\StegwareProject\\training_checkpoints\\4\\4'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    model.load_weights(checkpoint_path)

    def run(path):
        if os.path.isdir(path):
            time = datetime.now()
            files = glob.glob(path + "\\*\\*.jpg")
            c = 0
            for f in files:
                ds = prepare(f)
                res = model.predict(ds, verbose=1)
                print(f + ": " + str(res[0]))
                if res[0] > 0.5:
                    c += 1
            time = datetime.now()-time
            print(f"Time to Complete: {time.seconds} second(s)")
            print(c)
        else:
            time = datetime.now()
            ds = prepare(path)
            res = model.predict(ds, verbose=1)
            print(path + ": " + str(res[0]))
            time = datetime.now() - time
            print(f"Time to Complete: {time.seconds} second(s)")

    run_btn = tk.Button(button_frame, text='Run', background='#0ba6a3', foreground='#142c2b', width='12',
                        command=lambda: run(input_e.get()))
    run_btn.grid(column=1, row=0, sticky='s', padx=5, pady=5)

    window.mainloop()


def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    return img_array.reshape(-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)


if __name__ == '__main__':
    main()
