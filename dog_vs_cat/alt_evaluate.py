import sys
import json
import numpy as np
import progressbar
from typing import Type
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

sys.append("../")
from pyimagesearch.utils.ranked import rank5_accuracy
from dog_vs_cat.configs import dog_vs_cat_configs as configs
from pyimagesearch.preprocessing.croppreprocessor import CropPreprocessor
from pyimagesearch.preprocessing.meanpreprocessor import MeanPreprocessor
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor

# load RGB mean values
means = json.loads(open(configs.DATASET_MEAN).read())

# load preprocessors
ip = ImageToArrayPreprocessor()
cp = CropPreprocessor(width=227, height=227)
sp = SimplePreprocessor(width=227, height=227)
mp = MeanPreprocessor(rMean=means["R"], gMean=means["G"], bMean=means["B"])

# custom preprocessor class which utilizes custom preprocessors
class CustomPreprocessing:
    def __init__(self, preprocessors = []):
        self.preprocessors = preprocessors
    
    def preprocess(self, image):
        """ applies preprocessors on image """
        for preprocessor in self.preprocessors:
            image = preprocessor.preprocess(image)
        
        return image

# initialize preprocessor utilizers along with generator
test_preprocessor = CustomPreprocessing([sp, mp, ip])
test_aug = ImageDataGenerator(preprocessing_function=test_preprocessor.preprocess)
test_gen = test_aug.flow_from_directory(
    directory=configs.TEST_IMAGE,
    batch_size=configs.BATCH_SIZE
)

# load model and predict data from generator
model = load_model(configs.MODEL_PATH)
predictions = model.predict(test_gen, steps=test_gen.samples//configs.BATCH_SIZE)

# extract ground truth label from generator 
y_true = []
steps = len(test_gen) or (test_gen.samples // test_gen.batch_size)
for idx in np.arange(0, steps):
    (_, label_batch) = test_gen[idx]
    y_true.extend(label_batch)

# compute rank-1 accuracy
(rank1, _) = rank5_accuracy(y_true=y_true, y_pred=predictions)
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))

# re-initialize preprocessor utilizer along with generator
test_preprocessor = CustomPreprocessing([cp, ip])
test_aug = ImageDataGenerator(preprocessing_function=mp.preprocess)
test_gen = test_aug.flow_from_directory(
    directory=configs.TEST_IMAGE,
    batch_size=configs.BATCH_SIZE
)

# re-initialize ground truth label list
y_true = []
predictions = []

# number of unique batch image/labels in generator
steps = len(test_gen) # test_gen.samples // configs.BATCH_SIZE

# initialize progess bar
widgets = ["Evaluating: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(max_value=steps, widgets=widgets).start()

# loop over generator data in batch
for idx in np.arange(0, steps):
    
    # preprocess and predict image data in batches
    (image_batch, label_batch) = test_gen[idx]
    cropped_images = ([test_preprocessor.preprocess(img) for img in image_batch])
    predict_batch = model.predict(cropped_images, batch_size=configs.BATCH_SIZE)
    
    # stores ground truth labels and averaged predictions
    y_true.extend(label_batch)
    predictions.extend(np.mean(predict_batch, axis=0))
    pbar.update(idx)

pbar.finish()

# compute the rank-1 accuracy
print("[INFO] predicting on test data (with crops)...")
(rank1, _) = rank5_accuracy(y_true=y_true, y_pred=predictions)
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))