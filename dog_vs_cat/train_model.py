import sys
import h5py
import pickle
import argparse
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator

sys.path.append("../")
from pyimagesearch.nn.conv.bobonet import BoboNet
from dog_vs_cat.configs import dog_vs_cat_configs as configs
from pyimagesearch.io.hdf5datasetgenerator import HDF5DatasetGenerator

# define commandline arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-m",
    "--model",
    type=str,
    help="path to output model",
    required=True
)
ap.add_argument(
    "-c",
    "--classes",
    type=int,
    help="# of uniques labels/clases",
    default=getattr(configs, "NUM_CLASSES", 2)
)
ap.add_argument(
    "-e",
    "--epochs",
    type=int,
    help="# of epochs in model training",
    default=getattr(configs, "EPOCHS", 25)
)
ap.add_argument(
    "-bs",
    "--batch_size",
    type=int,
    help="# of features/labels per batch in training/test process",
    default=getattr(configs, "BATCH_SIZE", 32)
)
ap.add_argument(
    "-d",
    "--db",
    type=str,
    help="path to HDF database",
    default=getattr(configs, "EXTRACTED_FEATURES_HDF5", "../outputs/extracted_features.hdf5")
)

args = vars(ap.parse_args())

# dataset auggmentation
aug = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# dataset generator
train_gen = HDF5DatasetGenerator(
    aug=aug,
    binarize=True,
    db_path=args["db"],
    classes=args["classes"],
    feature_ref_name="data",
    batch_size=args["batch_size"]
)

# determine loss function based on unique number of classes
loss = "binary_crossentropy"
if args["classes"] > 2:
    loss = "categorical_crossentropy"

# configures optimizer and train model
opt = SGD(lr=1e-04, momentum=0.9, decay=1e-04/args["epoch"])
model = BoboNet.build(input_shape=(7 * 7 * 2048), classes=args["classes"])
model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
H = model.fit(
    train_gen.generate(),
    batch_size=args["batch_size"],
    validation_split=0.25,
    steps_per_epoch=train_gen.num_image // args["batch_size"]
)


# make and display a classification report
print("[INFO]: evaluating model...")
pred = model.predict(
    train_gen,
    batch_size=args["batch_size"],
    steps=train_gen.num_image // args["batch_size"]
)
print(classification_report(
    y_true=train_gen.db["labels"],
    y_pred=pred.argmax(axis=0),
    target_names=train_gen.db["class_labels"]
))


# compute the raw accuracy of the model
accuracy = accuracy_score(train_gen.db["labels"],  pred)
print(f"[INFO]: score: {accuracy}")

# saves the model to disk
print("[INFO]: saving model to disk...")
model.save(args["model"])

# close the database
train_gen.close()