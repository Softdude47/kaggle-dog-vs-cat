import sys
import argparse
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

plt.style.use("seaborn")
sys.path.append("../")
from pyimagesearch.nn.conv.bobonet import BoboNet
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
    default=2
)
ap.add_argument(
    "-e",
    "--epochs",
    type=int,
    help="# of epochs in model training",
    default=3
)
ap.add_argument(
    "-bs",
    "--batch-size",
    type=int,
    help="# of features/labels per batch in training/test process",
    default=32
)
ap.add_argument(
    "-train",
    "--train-db",
    type=str,
    help="path to HDF database",
    default="../datasets/kaggle_dog_vs_cat/hdf5/train_features.hdf5"
)
ap.add_argument(
    "-test",
    "--test-db",
    type=str,
    help="path to HDF database",
    default="../datasets/kaggle_dog_vs_cat/hdf5/test_features.hdf5"
)
ap.add_argument(
    "-val",
    "--val-db",
    type=str,
    help="path to HDF database",
    default="../datasets/kaggle_dog_vs_cat/hdf5/val_features.hdf5"
)

args = vars(ap.parse_args())

# dataset generator
print("[INFO] Loading datasets...")
train_gen = HDF5DatasetGenerator(
    binarize=True,
    db_path=args["train_db"],
    classes=args["classes"],
    feature_ref_name="data",
    batch_size=args["batch_size"]
)
test_gen = HDF5DatasetGenerator(
    binarize=True,
    db_path=args["test_db"],
    classes=args["classes"],
    feature_ref_name="data",
    batch_size=args["batch_size"]
)
val_gen = HDF5DatasetGenerator(
    binarize=True,
    db_path=args["val_db"],
    classes=args["classes"],
    feature_ref_name="data",
    batch_size=args["batch_size"]
)

# determine loss function based on unique number of classes
loss = "binary_crossentropy"
if args["classes"] > 2:
    loss = "categorical_crossentropy"

# configures optimizer and train model
print("[INFO] Training model...")
opt = SGD(lr=1e-04, momentum=0.9, decay=1e-04/args["epochs"])
model = BoboNet.build(input_shape=(7 * 7 * 2048, ), classes=args["classes"])
model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
H = model.fit(
    train_gen.generate(),
    epochs=args["epochs"],
    validation_data=val_gen.generate(),
    validation_steps=val_gen.num_images // args["batch_size"],
    steps_per_epoch=train_gen.num_images // args["batch_size"]
)

# make and display a classification report
print("[INFO]: Evaluating model on test dataset...")
pred = model.predict(
    test_gen.generate(),
    steps=test_gen.num_images // args["batch_size"]
)
pred = np.argmax(pred, axis=1)
print(classification_report(
    y_pred=pred,
    y_true=test_gen.db["labels"],
    target_names=test_gen.db["class_labels"]
))

# compute the raw accuracy with extra precision
acc = accuracy_score(y_true=test_gen.db["labels"], y_pred=pred)
print(f"[INFO] score: {acc}")

# saves the model to disk
print("[INFO]: saving model to disk...")
model.save(args["model"])

# close the database
train_gen.close()

# model train history
H = H.history
N = np.arange(0, len(H["loss"]))

# creates figures and subplots and plot
# models loss and accuracy history
(fig, axis) = plt.subplots(nrows=2, ncols=2, sharex=True)
(train_axis, val_axis) = axis

train_axis[0].plot(N, H["loss"], label="loss")
val_axis[0].plot(N, H["val_loss"], label="val_loss")

train_axis[1].plot(N, H["accuracy"], label="accuracy")
val_axis[1].plot(N, H["val_accuracy"], label="val_accuracy")

train_axis[0].set_ylabel("loss")
train_axis[1].set_ylabel("accuracy")

val_axis[0].set_ylabel("val_loss")
val_axis[1].set_ylabel("val_accuracy")

val_axis[0].set_xlabel("Epoch #")
val_axis[1].set_xlabel("Epoch #")

plt.tight_layout()

# show plot
plt.show()