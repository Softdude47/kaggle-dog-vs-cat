import cv2
import os

IMAGE_PATH = "../datasets/kaggle_dog_vs_cat/train/"

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
INTERPOLATION = cv2.INTER_AREA

TEST_SPLIT = 0.20
VAL_SPLIT = 0.20

HDF5_BUFFER_SIZE = 1000

TRAIN_HDF5 = "../datasets/kaggle_dog_vs_cat/hdf5/train.hdf5"
TEST_HDF5 = "../datasets/kaggle_dog_vs_cat/hdf5/test.hdf5"
VAL_HDF5 = "../datasets/kaggle_dog_vs_cat/hdf5/val.hdf5"

OUTPUT_PATH = "../outputs"
DATASET_MEAN = f"{OUTPUT_PATH}/dog_vs_cat.json"
MODEL_PATH = f"{OUTPUT_PATH}/dog_vs_cat.model"
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH, exist_ok=True)

# function used for extracting label from image path
def get_label(path:str):
    """ when file path is structured like '/directories/classname.index.jpg' """
    filename = path.split("/")[-1]
    label = filename.split(".")[0]
    return label