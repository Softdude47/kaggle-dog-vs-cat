import sys
import argparse
import progressbar
import numpy as np
from random import shuffle
from imutils.paths import list_images
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

sys.path.append("../")
from dog_vs_cat.configs import dog_vs_cat_configs as configs
from pyimagesearch.io.hdf5datasetwriter import HDF5DatasetWriter

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-d",
    "--dataset",
    help="path to input dataset",
    default=getattr(configs, "IMAGE_PATH", getattr(configs, "RAW_IMAGE", None))
)
ap.add_argument(
    "-o",
    "--output",
    help="path to output HDF5 file",
    default=getattr(configs, "EXTRACTED_FEATURES_HDF5", "../outputs/extracted_features.hdf5")
)
ap.add_argument(
    "-bs",
    "--batch-size",
    type=int,
    help="batch size of images to be passed through network",
    default=getattr(configs, "BATCH_SIZE", 128)
)
ap.add_argument(
    "-buff",
    "--buffer-size",
    type=int,
    help="size of feature extraction buffer",
    default=getattr(configs, "BUFFER_SIZE", 1000)
)
args = vars(ap.parse_args())
bs = args["batch_size"]

# load and shuffle image paths
train_path = list(list_images(args["dataset"]))
shuffle(train_path)

# extract and encode image labels
le = LabelEncoder()
train_labels = [configs.get_label(path) for path in train_path]
train_labels = le.fit_transform(train_labels)

# initialize hdf5 database class and add class names (string format) of images
db = HDF5DatasetWriter(args["output"], feature_ref_name="data", buffer_size=args["buffer_size"], shape=(len(train_path), 7 * 7 * 2048))
db.store_class_label(le.classes_)

# initialize model(without the head) and progressbar
model = ResNet50(weights="imagenet", include_top=False)
widgets = ["[INFO]: Extracting Features: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(max_value=len(train_path), widgets=widgets)

# loop over image paths in batches
for idx in np.arange(0, len(train_path), bs):
    
    # initialize image label and path batch
    # along side list of preprocessed images
    batch_path = train_path[idx: idx + bs]
    batch_labels = train_labels[idx: idx + bs]
    batch_images = []
    
    # loops over each image path
    for path in batch_path:
        # load actual image from path
        img = load_img(path=path, target_size=(224,224))
        img = img_to_array(img)
        
        # preprocess image
        img = np.expand_dims(img, axis=0)
        img = imagenet_utils.preprocess_input(img)
        
        # adds image to list
        batch_images.append(img)
        
    # vertically stack-up preprocessed images
    batch_images = np.vstack(batch_images)
    
    # extract and flatten image features
    features = model.predict(batch_images, batch_size=bs)
    features = np.reshape(features, (features.shape[0], 7 * 7 * 2048))
    
    # adds features and labels to database
    db.add(features, batch_labels)
    
    # updates progressbar
    pbar.update(idx)
    
# closes database and stops progressbar
db.close()
pbar.finish()