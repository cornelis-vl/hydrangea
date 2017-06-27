#!/usr/local/bin/python3

# Imports
import glob
import os
import time
import sys
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

# Ignore Keras updates
import warnings
warnings.filterwarnings("ignore")


# Functions

# I/O, ETL and other process functions.

def get_im_cv2(path, resize_to=(32,32)):
    """Get image using the image path and resize to (32 x 32) by default."""
    input_img = cv2.imread(path)
    resized_img = cv2.resize(input_img, resize_to, cv2.INTER_LINEAR)

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


def image_loader(folder, subset=100, resize_dim=(32,32), add_target=True):
    """Loads all images to array.
    Selects all available images unless subset is set to value between 1-99.
    """

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
        img = get_im_cv2(img_path, resize_to=resize_dim)
        imgs_collector.append(img)
        img_ids.append(file_base)
        process_no += 1

        if process_no % 100 == 0:
            print('Finished reading image {}..'.format(process_no))

    print('Total process time: {} seconds'.format(round(time.time() - start_time, 2)))

    # Create target variable for the selected images in the correct order.
    if add_target:
        target, target_ids = add_labels(dir=folder, ids=img_ids)

    elif not add_target:
        target = None
        target_ids = img_ids

    return imgs_collector, target_ids, target


def etl_images(image_loc, subset=100, resize_to=(32,32), train=True):
    features_raw, ids, target_raw = image_loader(folder=image_loc, subset=subset,
                                                 resize_dim=resize_to, add_target=train)

    # To array, transpose to fit model, and standardized for max. pixel value (255).
    X_train = np.array(features_raw, dtype=np.uint8)
    X_train = X_train.transpose((0, 3, 1, 2))
    X_train = X_train.astype('float32')
    X_train = X_train / 255

    # To array and categorical (one column for each class, 2 columns here)

    if train:
        Y_train = np.array(target_raw, dtype=np.uint8)
        Y_train = np_utils.to_categorical(Y_train, 2)
    elif not train:
        Y_train = None

    X_train_id  = np.array(ids, dtype=np.uint8)

    return X_train, X_train_id, Y_train


# General functions
def to_class(categorical_obj):
    """Transforms categorical object to a list where each value represents
    the selected class with maximum predicted probability.
    """
    categorical_list = list(categorical_obj)
    class_obj = []

    for yy in categorical_list:
        class_obj += [np.where(yy==max(yy))[0][0]]

    return class_obj


def train_test_split(features, target, identifier, prop=0.7):
    """Split full dataset in train and test data according to specified property
    assigned to the train data.
    """

    # Shuffle rows to mimic random selection
    rows = np.arange(0, target.shape[0])
    np.random.shuffle(rows)

    # Define threshold to split shuffled rows into train and test sets
    threshold = int(prop * len(rows))
    train_rows = rows[0:threshold]
    test_rows = rows[threshold:len(rows)]

    # Create train and test sets for features, target and ids
    train_id = identifier[train_rows]
    test_id = identifier[test_rows]
    train_features = features[train_rows]
    test_features = features[test_rows]
    train_target = target[train_rows]
    test_target = target[test_rows]

    return train_features, test_features, train_target, test_target, train_id, test_id


def cov_neural_net(input, output):
    """Store all model settings in one function"""

    input_layer_shape = input.shape[1:4]
    output_layer_shape = output.shape[1]


    # Convolution part
    model = Sequential()

    # Zero padding adds a row of zeros to top/bottom and column of zeros to left/right of features
    model.add(ZeroPadding2D((1, 1), input_shape=input_layer_shape, dim_ordering='th'))

    # 1-st convolution: This convolution returns an output of (8x8) using a (3x3) filter
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th', init= 'lecun_uniform', subsample=(1,1)))

    # Add zero padding to convoluted layer
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))

    # 2-nd convlution: Apply convolution again, with dropout and pool results
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th', init= 'lecun_uniform', subsample=(1,1)))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

    # 3-rd convolution: Same as above, different dimension
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(16, 3, 3, activation='relu', dim_ordering='th', init= 'lecun_uniform', subsample=(1,1)))

    # 4-th convulation: Same as above
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(16, 3, 3, activation='relu', dim_ordering='th', init= 'lecun_uniform', subsample=(1,1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    model.add(Dropout(0.2))

    # Neural network part

    # Input layer
    model.add(Flatten())
    model.add(Dense(32, activation='relu', init= 'lecun_uniform'))
    model.add(Dropout(0.4))

    # Hidden layer
    model.add(Dense(32, activation='relu', init= 'lecun_uniform'))
    model.add(Dropout(0.2))

    # Output for 2 classes
    model.add(Dense(output_layer_shape, activation='softmax'))

    # Optimizer (stochastic gradient descent)
    sgd = SGD(lr=1e-2, decay=1e-4, momentum=0.9, nesterov=False)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model

# Run script

if __name__ == '__main__':

    PROJECT_DIR = os.getcwd()  # locate to the 'hydrangea' folder
    DATA = '/data'

    train_dir = PROJECT_DIR + DATA + '/train'
    test_dir = PROJECT_DIR + DATA + '/test'

    # Load images
    subset_input = int(sys.argv[1])
    folder_input = sys.argv[2]

    if folder_input == 'all':
        train_dir = PROJECT_DIR + DATA + '/train'
    elif folder_input == 'preview':
        train_dir = PROJECT_DIR + DATA + '/train_preview'
    elif folder_input == 'submission':
        train_dir = PROJECT_DIR + DATA + '/train'
        test_dir = PROJECT_DIR + DATA + '/test'



    X_train, X_train_id, Y_train = etl_images(image_loc=train_dir, subset=subset_input,
                                              resize_to=(128,128), train=True)

    # Split to train and test datasets
    X_train_train, X_train_test, Y_train_train,\
    Y_train_test, ID_train_train, ID_train_test = train_test_split(features=X_train,
                                                                   target=Y_train,
                                                                   identifier=X_train_id)
    # Fit model
    clf = cov_neural_net(X_train_train, Y_train_train)

    clf.fit(X_train_train, Y_train_train,
               batch_size=32, nb_epoch=10, verbose=1,
               validation_data=(X_train_test, Y_train_test))

    # Predict train (for fit) and test (for predictive quality)
    Yp_train_train = clf.predict(X_train_train, batch_size=32, verbose=2)
    Yp_train_test = clf.predict(X_train_test, batch_size=32, verbose=2)

    # Convert for evaluation
    Yp_train_train_cat = to_class(Yp_train_train)
    Y_train_train_cat = to_class(Y_train_train)

    Yp_train_test_cat = to_class(Yp_train_test)
    Y_train_test_cat = to_class(Y_train_test)

    print("Fit quality of the model..")
    print(classification_report(Y_train_train_cat, Yp_train_train_cat))

    print("Predictive quality of the model..")
    print(classification_report(Y_train_test_cat, Yp_train_test_cat))

    if folder_input == 'submission':
        X_test, X_test_id, __ = etl_images(image_loc=test_dir, subset=100,
                                                  resize_to=(128, 128), train=False)

        Y_test_prop = clf.predict(X_test, batch_size=32, verbose=2)
        Y_test_cat = to_class(Y_test_prop)

        prediction = pd.DataFrame({'name': X_test_id,
                                   'invasive': Y_test_prop[:,1]})

        prediction.set_index('name', drop=True, inplace=True)
        prediction['invasive'] = round(prediction['invasive'], 3)
        prediction.sort_index(inplace=True)

        prep_date = time.strftime("%y%m%d_%H%M")
        filename = 'submission_{}.csv'.format(prep_date)

        prediction.to_csv(filename, sep=',', header=True, index=True)
