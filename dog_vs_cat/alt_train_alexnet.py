import os
import sys
import json
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# add project to path
sys.path.append("../")
from pyimagesearch.nn.conv.alexnet import AlexNet
from dog_vs_cat.configs import dog_vs_cat_configs as configs
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from pyimagesearch.io.simpledatasetloader import SimpleDatasetLoader
from pyimagesearch.preprocessing.meanpreprocessor import MeanPreprocessor
from pyimagesearch.preprocessing.patchpreprocessor import PatchPreprocessor
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor

# batch and epoch for training model
EPOCH = 75
BATCH_SIZE = 128

# mean of image channels
means = json.loads(open(configs.DATASET_MEAN).read())

# load preprocessors
iap = ImageToArrayPreprocessor()
mp = MeanPreprocessor(rMean=means["R"], gMean=means["G"], bMean=means["B"])
pp = PatchPreprocessor(width=227, height=227)
sp = SimplePreprocessor(width=227, height=227)

# custom preprocessor class which make use
# of the preprocessors
class CustomPreprocessing:
    def __init__(self, preprocessors = []):
        self.preprocessors = preprocessors
    
    def preprocess(self, image):
        """ applies preprocessors on image """
        for preprocessor in self.preprocessors:
            image = preprocessor.preprocess(image)
        
        return image

# initialize
train_preprocessors = CustomPreprocessing([pp, mp, iap])
val_preprocessors = CustomPreprocessing([sp, mp, iap])

# dataset generator
aug = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.15,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode="nearest",
    horizontal_flip=True,
    preprocessing_function=train_preprocessors.preprocess
)

train_generator = aug.flow_from_directory(
    directory=configs.TRAIN_IMAGE,
    target_size=(227, 227),
    class_mode="binary",
    batch_size=BATCH_SIZE
)
val_generator = aug.flow_from_directory(
    directory=configs.TEST_IMAGE,
    target_size=(227, 227),
    class_mode="binary",
    batch_size=BATCH_SIZE
)

# callback
path = os.path.sep.join([configs.OUTPUT_PATH, f"{os.getpid()}.png"])
callbacks = [TrainingMonitor(plot_path=path), ]

# initialize optimizer and model
opt = Adam(lr=1e-3)
model = AlexNet.build(width=227, height=227, classes=configs.NUM_CLASSES)

# compile and train model
model.compile(opt, loss="binary_crossentropy", metrics=["accuracy"])
model.fit(
    train_generator,
    epochs=EPOCH,
    steps_per_epoch=train_generator.samples//BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples//BATCH_SIZE,
    max_queue_size=BATCH_SIZE *2,
    callbacks=callbacks,
)

# save model to disk
model.save(configs.MODEL_PATH)