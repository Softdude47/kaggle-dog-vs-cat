import os
import cv2
import json
import numpy as np
import progressbar
from .configs import configs
from imutils.paths import list_images
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from pyimagesearch.io.HDF5DatasetWriter import HDF5DatasetWriter
from pyimagesearch.preprocessing.AspectAwarePreprocessor import AspectAwarePreprocessor

# split dataset into training, validation and testing
# using the class(labels) and path to image(features)
train_path = list(list_images(configs.IMAGE_PATH))
train_label = [path.split(os.sep)[-2] for path in train_path]
train_label = LabelEncoder().fit_transform(train_label)

(train_path, test_path), (train_label, test_label) = train_test_split(train_path, train_label, test_size=configs.TEST_SPLIT, stratify=train_label, random_state=42)
(train_path, val_path), (train_label, val_label) = train_test_split(train_path, train_label, test_size=configs.VAL_SPLIT, stratify=train_path, random_state=42)

datasets = [
    ("train", train_path, train_label, configs.TRAIN_HDF5),
    ("test", test_path, test_label, configs.TEST_HDF5),
    ("val", val_path, val_label, configs.VAL_HDF5),
]

# initialize the array to store training dataset's image channel
# and also initialize the preprocessor to use on all image
(R, G, B) = ([], [], [])
aap = AspectAwarePreprocessor(width=configs.IMAGE_WIDTH, height=configs.IMAGE_HEIGHT, interpolation=configs.INTERPOLATION)

# loop over each dataset group(train, val, test)
for (d_type, paths, labels, output_path) in datasets:
    
    # construct image shape and initialize the database
    input_shape = (len(paths), configs.IMAGE_HEIGHT, configs.IMAGE_WIDTH)
    db = HDF5DatasetWriter(name=output_path, feature_ref_name="data", buffer_size=configs.HDF5_BUFFER_SIZE, shape=input_shape)
    
    # displays progress
    print(f"[INFO] building {output_path}")
    widgets = ["Building dataset: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(max_value=len(paths), widgets=widgets)
    
    # loop over the datasets image path and the corresponding label
    for (i, (path, label)) in enumerate(zip(output_path, labels)):
        # read image from path and apply preprocessor
        image = cv2.imread(path)
        image = aap.preprocess(image)
        
        # extract the image channel from training dataset image
        # appends it to its respective array
        if d_type == "train":
            (b, g, r) = image
            R.append(r)
            G.append(g)
            B.append(b)
        
        # add the preprocessed image along with its
        # label to database and update progess display
        db.add([image, ], [label, ])
        pbar.update(i)
        
    # close database
    db.close()
    pbar.finish()

# stores the mean of training dataset image channel to json file
mean_dict = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(configs.DATASET_MEAN)
f.write(json.dumps(mean_dict))
f.close()