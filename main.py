# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

import cv2
import matplotlib.pyplot as plt
import keras
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, BatchNormalization
from keras.models import Sequential
from keras.models import load_model
from sklearn.tree import DecisionTreeRegressor
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import time
import tensorflow_hub as hub
from keras.preprocessing import image

dirname = 'C:/Users/melec/PycharmProjects/Chessman-image-dataset/Chess'
dir_chess_folders = os.listdir(dirname)
dir_chess_paths = [os.path.join(dirname, path) for path in dir_chess_folders]

dirname_work = 'C:/Users/melec/PycharmProjects'
dir_work = os.path.join('C:/Users/melec/PycharmProjects', 'Result')
dir_work_chess = os.path.join(dir_work, 'Chess')

def check_file(model):
    sunflower_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f1/Chess_piece_-_Black_king.JPG/100px-Chess_piece_-_Black_king.JPG"
    sunflower_path = tf.keras.utils.get_file('krul', origin=sunflower_url)

    img = tf.keras.utils.load_img(
        sunflower_path, target_size=(300, 300)#, color_mode="grayscale"
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

def makefolders():
    os.mkdir('C:/Users/melec/PycharmProjects/Result/Chess')
    os.mkdir('C:/Users/melec/PycharmProjects/Result/Chess/Rook')
    os.mkdir('C:/Users/melec/PycharmProjects/Result/Chess/Knight')
    os.mkdir('C:/Users/melec/PycharmProjects/Result/Chess/Queen')
    os.mkdir('C:/Users/melec/PycharmProjects/Result/Chess/Pawn')
    os.mkdir('C:/Users/melec/PycharmProjects/Result/Chess/King')
    os.mkdir('C:/Users/melec/PycharmProjects/Result/Chess/Bishop')
    bishop_path_work = os.path.join(dir_work_chess, 'Bishop')
    knight_path_work = os.path.join(dir_work_chess, 'Knight')
    queen_path_work = os.path.join(dir_work_chess, 'Queen')
    rook_path_work = os.path.join(dir_work_chess, 'Rook')
    king_path_work = os.path.join(dir_work_chess, 'King')
    pawn_path_work = os.path.join(dir_work_chess, 'Pawn')
    dir_chess_folders_work = os.listdir(dir_work_chess)
    dir_chess_paths_work = [os.path.join(dir_work_chess, path) for path in dir_chess_folders_work]


    def image_binarization(path_from, path_to):    #wyszarzanie

        i=1
        files = os.listdir(path_from)
        for file in files:
            try:
                file_dir = os.path.join(path_from, file)
                file_dir_save = os.path.join(path_to, file)
                img = Image.open(file_dir)
                img = img.convert("1")
                img.save(file_dir_save)
                i=i+1
            except:
                continue

    image_binarization(dir_chess_paths[0], bishop_path_work)
    image_binarization(dir_chess_paths[1], king_path_work)
    image_binarization(dir_chess_paths[2], rook_path_work)
    image_binarization(dir_chess_paths[3], pawn_path_work)
    image_binarization(dir_chess_paths[4], queen_path_work)
    image_binarization(dir_chess_paths[5], knight_path_work)


    def image_median_filtering(path_from, path_to, window_size=3):  #Polepszanie jakości szarego zdjęcia

        i=1
        files = os.listdir(path_from)
        for file in files:
            try:
                file_dir = os.path.join(path_from, file)
                file_dir_save = os.path.join(path_to, file)
                img = Image.open(file_dir)
                img = img.filter(ImageFilter.MedianFilter(window_size))
                img.save(file_dir_save)
                i=i+1
            except:
                continue


    image_median_filtering(bishop_path_work, bishop_path_work)
    image_median_filtering(king_path_work, king_path_work)
    image_median_filtering(rook_path_work, rook_path_work)
    image_median_filtering(pawn_path_work, pawn_path_work)
    image_median_filtering(queen_path_work, queen_path_work)
    image_median_filtering(knight_path_work, knight_path_work)

batch_size = 32
img_height = 300
img_width = 300

train_ds = tf.keras.utils.image_dataset_from_directory(
    dir_work_chess,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dir_work_chess,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names

print(class_names)

def train_model():
    batch_size = 32
    img_height = 300
    img_width = 300

    train_ds = tf.keras.utils.image_dataset_from_directory(
        dir_work_chess,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dir_work_chess,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    normalization_layer = layers.Rescaling(1. / 255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]

    print(np.min(first_image), np.max(first_image))

    num_classes = len(class_names)

    model = Sequential([
        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    epochs = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    model.save('saved_model/my_model')
    check_file(model)


if not os.path.exists(dir_work_chess):
    makefolders()

model_path = "C:/Users/melec/PycharmProjects/BIAI/saved_model/my_model.pb"
want_to_train_model : bool = 0
file_or_folder : bool = 1

if want_to_train_model == 1:
    train_model()
else:
    model = tf.keras.models.load_model('saved_model/my_model')
    check_file(model)








