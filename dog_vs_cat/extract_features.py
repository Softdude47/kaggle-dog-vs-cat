import sys
import argparse
import progressbar
import numpy as np
from imutils.paths import list_images
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
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
    required=True
)
ap.add_argument(
    "-train",
    "--train-path",
    help="path to output HDF5 file(training)",
    default="../datasets/kaggle_dog_vs_cat/hdf5/train_features.hdf5"
)
ap.add_argument(
    "-val",
    "--val_path",
    help="path to output HDF5 file(train validation)",
    default="../datasets/kaggle_dog_vs_cat/hdf5/val_features.hdf5"
)
ap.add_argument(
    "-test",
    "--test_path",
    help="path to output HDF5 file(evaluation)",
    default="../datasets/kaggle_dog_vs_cat/hdf5/test_features.hdf5"
)
ap.add_argument(
    "-split",
    "--test-size",
    help="size of test and validation dataset",
    default=6000
)
ap.add_argument(
    "-bs",
    "--batch-size",
    type=int,
    help="batch size of images to be passed through network",
    default=128
)

args = vars(ap.parse_args())
bs = args["batch_size"]
test_size = float(args["test_size"])

# load and shuffle image paths
train_path = list(list_images(args["dataset"]))
train_labels = [configs.get_label(path) for path in train_path]

(train_path, val_path, train_labels, val_labels) = train_test_split(
    train_path,
    train_labels,
    test_size=test_size/2,
)
(train_path, test_path, train_labels, test_labels) = train_test_split(
    train_path,
    train_labels,
    test_size=test_size/2,
)



# construct list of dataset split
DATASET = [
    (train_labels, train_path, args["train_path"]),
    (test_labels, test_path, args["test_path"]),
    (val_labels, val_path, args["val_path"])
]

# initialize model without the head
model = ResNet50(weights="imagenet", include_top=False)

for (label, input_path, output_path) in DATASET:
    
    # start progressbar
    print(f"[INFO]: building {output_path}...")
    widgets = ["[INFO]: Extracting Features: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(max_value=len(input_path), widgets=widgets).start()

    # encoding labels
    le = LabelEncoder()
    le.fit(label)
    label = le.transform(label)
    
    # initialize hdf5 database class and add class names (string format) of images
    db = HDF5DatasetWriter(file_name=output_path, feature_ref_name="data", buffer_size=10000, shape=(len(input_path), 7 * 7 * 2048))
    db.store_class_label(le.classes_)
    
    # loop over image paths in batches
    for idx in np.arange(0, len(input_path), bs):
        
        # initialize image label and path batch
        # along side list of preprocessed images
        batch_path = input_path[idx: idx + bs]
        batch_labels = label[idx: idx + bs]
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