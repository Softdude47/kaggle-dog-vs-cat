import sys
import json
import numpy as np
import progressbar
from keras.models import load_model

sys.append("../")
from pyimagesearch.utils.ranked import rank5_accuracy
from dog_vs_cat.configs import dog_vs_cat_configs as configs
from pyimagesearch.io.hdf5datasetgenerator import HDF5DatasetGenerator
from pyimagesearch.preprocessing.croppreprocessor import CropPreprocessor
from pyimagesearch.preprocessing.meanpreprocessor import MeanPreprocessor
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor

# load the RGB mean values
means = json.loads(open(configs.DATASET_MEAN).read())

# initialize the preprocessors
ip = ImageToArrayPreprocessor()
cp = CropPreprocessor(width=227, height=227)
sp = SimplePreprocessor(width=227, height=227)
mp = MeanPreprocessor(rMean=means["R"], gMean=means["G"], bMean=means["B"])

# initialize testing datasets generation
test_gen = HDF5DatasetGenerator(
    configs.TEST_HDF5,
    feature_ref_name="data",
    batch_size=configs.BATCH_SIZE,
    preprocessors=[sp, mp, ip]
)

# load model
model = load_model(configs.MODEL_PATH)

# predict test generator
predictions = model.predict(test_gen.generate(passes=1), steps=test_gen.num_images//configs.BATCH_SIZE)

# compute and display rank-1 accuracy
(rank1, _) = rank5_accuracy(y_true=test_gen.db["labels"], y_pred=predictions)
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))

# close generator
test_gen.close()

# initialize progressbar for new model evaluation
widgets = ["Evaluating: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(widgets=widgets, max_value=test_gen.num_images//configs.BATCH_SIZE).start()

# re-initialize generator and prediction
# list and loop over batches of data in generator
test_gen = HDF5DatasetGenerator(
    configs.TEST_HDF5,
    feature_ref_name="data",
    batch_size=configs.BATCH_SIZE,
    preprocessors=[sp, mp, ip]
)
predictions = []
for (i, (images, label)) in enumerate(test_gen.generate(passes=1)):
    for image in images:
        # generate ten cropped copies of each images
        # and appl image-to-array preprocessor
        cropped_images = cp.preprocess(image)
        cropped_image = np.array([ip.preprocess(crop) for crop in cropped_images])
        
        # predicts the image class and append
        # the average predictions to prediction list
        pred = model.predict(cropped_images)
        predictions.append(np.mean(pred, axis=0))
    pbar.update(i)

# compute the rank-1 accuracy
pbar.finish()
print("[INFO] predicting on test data (with crops)...")
(rank1, _) = rank5_accuracy(y_true=test_gen.db["labels"], y_pred=predictions)
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
test_gen.close()