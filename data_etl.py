import os
import cv2
import time
import glob
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils

HOME_DIR = '/Users/cornelisvletter/Desktop/Programming/Personal'
PROJECT_FOLDER = '/Kaggle/hydrangea'


MODEL = '/model'

#To data
DATA = '/data'

os.chdir(HOME_DIR + PROJECT_FOLDER + DATA)

def get_im_cv2(path):
    img = cv2.imread(path)
    resized_img = cv2.resize(img, (32, 32), cv2.INTER_LINEAR)
    return resized_img


def add_labels(dir, ids):
    label_file = glob.glob1(dir, '*.csv')[0]
    file_path = dir + '/' + label_file
    labels = pd.read_csv(file_path, sep=',')
    id_df = pd.DataFrame(ids)
    id_df.columns = ['id']

    target = id_df.merge(
        labels,
        how='left',
        left_on='id',
        right_on='name',
        indicator=True,
        copy=True)

    y_train = target['invasive']
    x_train_id = target['id']

    return y_train, x_train_id


def load_images(folder, subset=100):
    x_train = []
    img_ids = []

    start_time = time.time()

    print('Read train images')
    files = glob.glob1(folder, '*.jpg')

    subset_size = int((subset/100) * len(files))

    for fl in files[:subset_size]:
        flbase = int(os.path.basename(fl)[:-4])
        img_path = folder + '/' + fl
        img = get_im_cv2(img_path)
        x_train.append(img)
        img_ids.append(flbase)
        if len(img_ids) % 100 == 0:
            print('Finished reading image {}.'.format(len(img_ids)))

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))

    y_train, x_train_id = add_labels(dir=folder, ids=img_ids)

    return x_train, x_train_id, y_train


train_dir = HOME_DIR + PROJECT_FOLDER + DATA + '/train'
test_dir = HOME_DIR + PROJECT_FOLDER + DATA + '/test'

x5_train, x5_train_id, y5_train = load_images(folder=train_dir, subset=5)

X_train = x5_train
X_train_id = x5_train_id
Y_train = y5_train

def etl():
    X_train, X_train_id, Y_train = load_images(folder=train_dir, subset=5)

    X_train = np.array(X_train, dtype=np.uint8)
    X_train = X_train.astype('float32')
    X_train = X_train / 255

    Y_train = np.array(Y_train, dtype=np.uint8)
    Y_train = np_utils.to_categorical(Y_train, 2)

    X_train_id  = np.array(X_train_id, dtype=np.uint8)

    return X_train, X_train_id, Y_train

X_train, X_train_id, Y_train = etl()

def create_model():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 32, 32), dim_ordering='th'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th', init= 'lecun_uniform', subsample=(1,1)))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th', init= 'lecun_uniform', subsample=(1,1)))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(16, 3, 3, activation='relu', dim_ordering='th', init= 'lecun_uniform', subsample=(1,1)))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(16, 3, 3, activation='relu', dim_ordering='th', init= 'lecun_uniform', subsample=(1,1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(32, activation='relu', init= 'lecun_uniform'))
    model.add(Dropout(0.4))
    model.add(Dense(32, activation='relu', init= 'lecun_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation='softmax'))

    sgd = SGD(lr=1e-2, decay=1e-4, momentum=0.9, nesterov=False)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model

