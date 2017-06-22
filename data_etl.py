import os
import cv2
import time
import glob
import pandas as pd
from keras.models import Sequential

HOME_DIR = '/Users/cornelisvletter/Desktop/Programming/Personal'
PROJECT_FOLDER = '/Kaggle/hydrangea'


MODEL = '/model'

#To data
DATA = '/data'

os.chdir(HOME_DIR + PROJECT_FOLDER + DATA)

def get_im_cv2(path):
    img = cv2.imread(path)
    #resized = cv2.resize(img, (32, 32), cv2.INTER_LINEAR)
    return img

def add_labels(dir, ids):
    label_file = glob.glob1(dir, '*.csv')[0]
    labels = pd.read_csv(label_file, sep=',')
    id_df = pd.DataFrame(ids)
    id_df.columns = ['id']

    y_train = id_df.merge(
        labels,
        how='left',
        left_on='id',
        right_on='name',
        indicator=True,
        copy=True)

    return y_train


def load_images(folder, subset=100):
    img_array = []
    img_ids = []

    start_time = time.time()

    print('Read train images')
    files = glob.glob1(folder, '*.jpg')

    subset_size = int((subset/100) * len(files))

    for fl in files[:subset_size]:
        flbase = int(os.path.basename(fl)[:-4])
        img = cv2.imread(fl)
        img_array.append(img)
        img_ids.append(flbase)
        if len(img_ids) % 100 == 0:
            print('Finished reading image {}.'.format(len(img_ids)))

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))

    return img_array, img_ids


train_dir = HOME_DIR + PROJECT_FOLDER + DATA + '/train'
test_dir = HOME_DIR + PROJECT_FOLDER + DATA + '/test'

tr_snip, trid_snip = load_images(train_dir, subset=5)

