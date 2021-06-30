import cv2
import os

# path to dataset
IMAGE_PATH = "../datasets/kaggle_dog_vs_cat/train/"

# image properties
BATCH_SIZE = 64
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
INTERPOLATION = cv2.INTER_AREA

# since we do not have validation data or access to the testing
# labels we need to take a number of images from the training
# data and use them instead
NUM_CLASSES = 2
TEST_SPLIT = 1250 * NUM_CLASSES
VAL_SPLIT = 1250 * NUM_CLASSES

# path to the output training, validation, and testing
# HDF5 files
HDF5_BUFFER_SIZE = 1000
HDF5_PATH = "../datasets/kaggle_dog_vs_cat/hdf5"
TRAIN_HDF5 = f"{HDF5_PATH}/train.hdf5"
TEST_HDF5 = f"{HDF5_PATH}/test.hdf5"
VAL_HDF5 = f"{HDF5_PATH}/val.hdf5"

# path to the output training, validation, and testing
# image files
RAW_IMAGE = "../datasets/kaggle_dog_vs_cat/image"
TRAIN_IMAGE = f"{RAW_IMAGE}/train"
TEST_IMAGE = f"{RAW_IMAGE}/test"
VAL_IMAGE = f"{RAW_IMAGE}/val"

# path to the output directory used for storing plots,
# classification reports, etc.
OUTPUT_PATH = "../outputs"

# path to dataset mean
DATASET_MEAN = f"{OUTPUT_PATH}/dog_vs_cat.json"

# path to output model file
MODEL_PATH = f"{OUTPUT_PATH}/dog_vs_cat.model"

# automatically create parent directories
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(RAW_IMAGE, exist_ok=True)
os.makedirs(HDF5_PATH, exist_ok=True)

# function used for extracting label from image path
def get_label(path:str):
    """ when file path is structured like '/directories/classname.index.jpg' """
    filename = path.split(os.path.sep)[-1]
    label = filename.split(".")[0]
    return label