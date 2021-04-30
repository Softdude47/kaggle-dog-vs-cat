import os
import cv2
import json
import progressbar
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from cat_vs_dog.configs import configs
from pyimagesearch.nn.conv.AlexNet import AlexNet
from pyimagesearch.callbacks.TrainingMonitor import TrainingMonitor
from pyimagesearch.io.HDF5DatasetGenerator import HDF5DatasetGenerator
from pyimagesearch.preprocessing.MeanPreprocessor import MeanPreprocessor
from pyimagesearch.preprocessing.PatchPreprocessor import PatchPreprocessor
from pyimagesearch.preprocessing.SimplePreprocessor import SimplePreprocessor
from pyimagesearch.preprocessing.ImageToArrayPreprocessor import ImageToArrayPreprocessor

EPOCH = 75
BATCH_SIZE = 128

means = json.loads(open(configs.DATASET_MEAN).read())

iap = ImageToArrayPreprocessor()
mp = MeanPreprocessor(rMean=means["R"], gMean=means["G"], bMean=means["B"])
pp = PatchPreprocessor(width=227, height=227)
sp = SimplePreprocessor(wdith=227, height=227)

aug = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.15,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode="nearest",
    horizontal_flip=True
)

train_generator = HDF5DatasetGenerator(
    db_path=configs.TRAIN_HDF5,
    batch_size=BATCH_SIZE,
    preprocessors=[pp, mp, iap],
    aug=aug,
    binarize=True,
    classes=2
)

val_generator = HDF5DatasetGenerator(
    db_path=configs.TEST_HDF5,
    batch_size=BATCH_SIZE,
    preprocessors=[sp, mp, iap],
    aug=None,
    binarize=True,
    classes=2
)

path = os.path.sep.join([configs.OUTPUT_PATH, f"{os.getpid()}.png"])
callbacks = [TrainingMonitor(plot_path=path)]

opt = Adam(lr=1e-3)
model = AlexNet().build(width=227, height=227)

model.compile(opt, loss="binary_crossentropy", metrics=["accuracy"])
model.fit(
    train_generator.generate(),
    epochs=EPOCH,
    steps_per_epoch=train_generator.num_images//BATCH_SIZE,
    validation_steps=val_generator.num_images//BATCH_SIZE,
    validation_data=val_generator.generate(),
    max_queue_size=BATCH_SIZE *2,
    callbacks=callbacks,
)

model.save(configs.MODEL_PATH)
train_generator.close()
val_generator.close()