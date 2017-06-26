# IMPORTS

import glob
import os
import time
import cv2
import numpy as np
import pandas as pd
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report

# FUNCTIONS

# I/O, ETL and other process functions.

def get_im_cv2(path, resize_dim=(32,32)):
    """Get image using the image path and resize to (32 x 32) by default."""
    input_img = cv2.imread(path)
    resized_img = cv2.resize(input_img, resize_dim, cv2.INTER_LINEAR)

    return resized_img


def show_img(img):
    """Simple wrap to show an image."""
    plt.imshow(img)


def add_labels(dir, ids):
    """In this case, the target values (and labels) should be added separately."""
    label_file = glob.glob1(dir, '*.csv')[0]
    file_path = dir + '/' + label_file
    labels = pd.read_csv(file_path, sep=',')
    id_df = pd.DataFrame(ids)
    id_df.columns = ['id']

    combined = id_df.merge(labels, how='left', left_on='id', right_on='name', indicator=True, copy=True)

    target = combined['invasive']
    target_id = combined['id']

    return target, target_id


def image_loader(folder, subset=100):
    "Loads all images to array. Selects all available images unless subset is set to value between 1-99."

    imgs_collector = []
    img_ids = []

    start_time = time.time()

    print('Reading images folder..')
    files = glob.glob1(folder, '*.jpg')

    # Shuffle list for random selection of images.
    np.random.shuffle(files)
    subset_size = int((subset/100) * len(files))

    print('Total of {} images to process..'.format(subset_size))
    process_no = 1

    for file in files[:subset_size]:

        file_base = int(os.path.basename(file)[:-4])
        img_path = folder + '/' + file
        img = get_im_cv2(img_path)
        imgs_collector.append(img)
        img_ids.append(file_base)
        process_no += 1

        if len(process_no) % 100 == 0:
            print('Finished reading image {}..'.format(process_no))

    print('Total process time: {} seconds'.format(round(time.time() - start_time, 2)))

    # Create target variable for the selected images in the correct order.
    target, target_ids = add_labels(dir=folder, ids=img_ids)

    return imgs_collector, target_ids, target


def etl_images(image_loc, subset=100):
    features_raw, ids, target_raw = image_loader(folder=image_loc, subset=subset)

    # To array, transpose to fit model, and standardized for max. pixel value (255).
    X_train = np.array(features_raw, dtype=np.uint8)
    X_train = X_train.transpose((0, 3, 1, 2))
    X_train = X_train.astype('float32')
    X_train = X_train / 255

    # To array and categorical (one column for each class, 2 columns here)
    Y_train = np.array(target_raw, dtype=np.uint8)
    Y_train = np_utils.to_categorical(Y_train, 2)

    X_train_id  = np.array(ids, dtype=np.uint8)

    return X_train, X_train_id, Y_train


# General functions
def to_class(categorical_obj):
    """Transforms categorical object to a list where each value represents
    the selected class with maximum predicted probability.
    """

    categorical_list = list(categorical_obj)

    class_obj = []

    for yy in categorical_obj:
        class_obj += [np.where(yy==max(yy))[0][0]]

    return class_obj

def train_test_split(features, target, identifier, p=0.7):
    rows = np.arange(0, target.shape[0])
    np.random.shuffle(rows)

    threshold = int(p * len(rows))
    train_rows = rows[0:threshold]
    test_rows = rows[threshold:len(rows)]
    train_id = identifier[train_rows]
    test_id = identifier[test_rows]

    train_features = features[train_rows]
    test_features = features[test_rows]

    train_target = target[train_rows]
    test_target = target[test_rows]

    return train_features, test_features, train_target, test_target, train_id, test_id

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
    model.add(Dense(2, activation='softmax'))

    sgd = SGD(lr=1e-2, decay=1e-4, momentum=0.9, nesterov=False)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model




train_dir = HOME_DIR + PROJECT_FOLDER + DATA + '/train'
test_dir = HOME_DIR + PROJECT_FOLDER + DATA + '/test'

X_train, X_train_id, Y_train = load_images(folder=train_dir, subset=25)

test_model = create_model()

X_train_train, X_train_test, Y_train_train,\
Y_train_test, ID_train_train, ID_train_test = train_test_split(features=X_train,
                                                               target=Y_train,
                                                               identifier=X_train_id)

test_model.fit(X_train_train, Y_train_train,
               batch_size=32, nb_epoch=10, verbose=1,
               validation_data=(X_train_test, Y_train_test))

Yp_train_train = test_model.predict(X_train_train, batch_size=32, verbose=2)
Yp_train_test = test_model.predict(X_train_test, batch_size=32, verbose=2)

Yp_train_train_cat = convert_to_class(Yp_train_train)
Y_train_train_cat = convert_to_class(Y_train_train)

print(classification_report(Y_train_train_cat, Yp_train_train_cat))
