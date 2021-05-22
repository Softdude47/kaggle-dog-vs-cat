import cv2
import os

IMAGE_PATH = "../datasets/kaggle_dog_vs_cat/train/"

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
INTERPOLATION = cv2.INTER_AREA

NUM_CLASSES = 2
TEST_SPLIT = 1250 * NUM_CLASSES
VAL_SPLIT = 1250 * NUM_CLASSES

HDF5_BUFFER_SIZE = 1000

HDF5_PATH = "../datasets/kaggle_dog_vs_cat/hdf5"
TRAIN_HDF5 = f"{HDF5_PATH}/train.hdf5"
TEST_HDF5 = f"{HDF5_PATH}/test.hdf5"
VAL_HDF5 = f"{HDF5_PATH}/val.hdf5"

IMAGE_PATH = "../datasets/kaggle_dog_vs_cat/image"
TRAIN_IMAGE = f"{IMAGE_PATH}/train"
TEST_IMAGE = f"{IMAGE_PATH}/test"
VAL_IMAGE = f"{IMAGE_PATH}/val"

OUTPUT_PATH = "../outputs"
DATASET_MEAN = f"{OUTPUT_PATH}/dog_vs_cat.json"
MODEL_PATH = f"{OUTPUT_PATH}/dog_vs_cat.model"


os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(HDF5_PATH, exist_ok=True)

# function used for extracting label from image path
def get_label(path:str):
    """ when file path is structured like '/directories/classname.index.jpg' """
    filename = path.split(os.path.sep)[-1]
    label = filename.split(".")[0]
    return label