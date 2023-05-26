# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os

import cv2
import nnabla as nn
import time
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
train_loss_results = []
train_accuracy_results = []

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

def trainModel(model, normalized_ds, optimizer):
    features, labels = next(iter(normalized_ds))

    loss_value, grads = grad(model, features, labels)

    num_epochs = 201

    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # Training loop - using batches of 32
        for x, y in train_ds:
            # Optimize the model
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            epoch_accuracy.update_state(y, model(x, training=True))

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        if epoch % 50 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                        epoch_loss_avg.result(),
                                                                        epoch_accuracy.result()))
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(train_loss_results)

    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(train_accuracy_results)
    plt.show()


def fitFunction(model, train_ds, optimizer):

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    epochs = 3
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Update training metric.
            train_acc_metric.update_state(y_batch_train, logits)

            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %d samples" % ((step + 1) * batch_size))

        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_ds:
            val_logits = model(x_batch_val, training=False)
            # Update val metrics
            val_acc_metric.update_state(y_batch_val, val_logits)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Time taken: %.2fs" % (time.time() - start_time))

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

    # tf.keras.Input(
    #     shape=None,
    #     batch_size=None,
    #     name=None,
    #     dtype=None,
    #     sparse=None,
    #     tensor=None,
    #     ragged=None,
    #     type_spec=None,
    # )

    model = Sequential([
        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(1, 2, padding='same', activation='relu'),
        layers.MaxPooling2D(), layers.BatchNormalization(),
        layers.Conv2D(8, 2, padding='same', activation='relu'),
        layers.MaxPooling2D(), layers.BatchNormalization(),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(), layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])


    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    # optimizer='adam'

    trainModel(model, normalized_ds, optimizer)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()

    # fitFunction(model, train_ds, optimizer)

    # epochs = 3
    # history = model.fit(
    #     train_ds,
    #     validation_data=val_ds,
    #     epochs=epochs
    # )

    lossList.append(history.history['loss'])
    accuracyList.append(history.history['accuracy'])
    valLossList.append(history.history['val_loss'])
    valAccuracyList.append(history.history['val_accuracy'])

    model.save('saved_model/my_model')
    check_file(model)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
def loss(model, x, y, training):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  y_ = model(x, training=training)

  return loss_object(y_true=y, y_pred=y_)
def test_step(x, y, model, val_acc_metric):
    val_logits = model(x, training=False)
    val_acc_metric.update_state(y, val_logits)
def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

def train_step(x, y, model):
    with tf.GradientTape() as tape:
        train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
        optimizer = keras.optimizers.SGD(learning_rate=1e-3)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
        # Add any extra losses created during the forward pass.
        loss_value += sum(model.losses)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, logits)
    return loss_value

if not os.path.exists(dir_work_chess):
    makefolders()

model_path = "C:/Users/melec/PycharmProjects/BIAI/saved_model/my_model.pb"
want_to_train_model : bool = 1
file_or_folder : bool = 0

if want_to_train_model == 1:
    for i in range(0, 1):
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








