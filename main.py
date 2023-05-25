# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os

import cv2
import keras.losses
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from PIL import Image, ImageFilter
from keras.models import Sequential
from tensorflow.keras import layers

from keras.layers import Dense
from keras.optimizers import Adam

lossList = []
accuracyList = []
valLossList = []
valAccuracyList = []

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

dirname = 'C:/Users/melec/PycharmProjects/Chessman-image-dataset/Chess'
dir_chess_folders = os.listdir(dirname)
dir_chess_paths = [os.path.join(dirname, path) for path in dir_chess_folders]

dirname_work = 'C:/Users/melec/PycharmProjects'
dir_work = os.path.join('C:/Users/melec/PycharmProjects', 'Result')
dir_work_chess = os.path.join(dir_work, 'Chess')

def show(photo_path, score):
    img = np.asarray((Image.open(photo_path)))
    plt.title("This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score)))
    plt.imshow(img)
    plt.show()

# def CV(model, photo_path):
#     image = cv2.imread(photo_path)
#     x = np.array([0.25, 0.25, 0.75, 0.75]).reshape(1, 4)
#     y = model.predict(x)
#     x1, y1, x2, y2 = y.flatten()
#     h, w, _ = image.shape
#
#     x1, y1, x2, y2 = int(w * x1), int(h * y1), int(w * x2), int(h * y2)
#     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
#     cv2.imshow('Image with Rectangle', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

def check_file(model):

    photo_path = 'C:/Users/melec/Desktop/wieza.jpg'
    #CV(model, photo_path)
    img = tf.keras.utils.load_img(
        photo_path, target_size=(img_height, img_width)
    )
    filename, file_extension = os.path.splitext(photo_path)
    save_path = "tmp" + file_extension
    img2 = img.convert("1")
    img2 = img2.filter(ImageFilter.MedianFilter(3))
    img2.save(save_path)

    img_final = tf.keras.utils.load_img(
        save_path, target_size=(img_height, img_width)
    )
    # img_final.show()

    img_array = tf.keras.utils.img_to_array(img_final)
    plt.imshow(img_array / 255.)

    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])


    #tf.keras.preprocessing.image.array_to_img(img).show()

    # show(photo_path, score)

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

    print(tf.keras.losses.Loss)

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

    AUTOTUNE = tf.data.AUTOTUNE #Dynamiczne dostosowywanie operacji wczytywania danych (buffor, wątki itp)

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE) #Przechowuje obrazy w pamięci
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE) #wykonywanie modelu podczas uczenia

    normalization_layer = layers.Rescaling(1. / 255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]

    # print(np.min(first_image), np.max(first_image))

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

    epochs = 3
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True) #różnica między dwoma entropiami

    lossList.append(history.history['loss'])
    accuracyList.append(history.history['accuracy'])
    valLossList.append(history.history['val_loss'])
    valAccuracyList.append(history.history['val_accuracy'])

    model.save('saved_model/my_model')
    check_file(model)


if not os.path.exists(dir_work_chess):
    makefolders()

model_path = "C:/Users/melec/PycharmProjects/BIAI/saved_model/my_model.pb"
want_to_train_model : bool = 1
file_or_folder : bool = 0

if want_to_train_model == 1:
    for i in range(0, 5):
        train_model()

    loss = np.mean(lossList)
    accuracy = np.mean(accuracyList)
    valLoss = np.mean(valLossList)
    valAccuracy = np.mean(valAccuracyList)

    print('loss', loss)
    print('accuracy', accuracy)
    print('valLos', valLoss)###sasa
    print('valAccuracy', valAccuracy)
else:
    model = tf.keras.models.load_model('saved_model/my_model')
    check_file(model)








