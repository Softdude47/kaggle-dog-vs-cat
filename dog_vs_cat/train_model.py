import sys
import h5py
import pickle
import argparse
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

sys.path.append("../")
from dog_vs_cat.configs import dog_vs_cat_configs as configs

# define commandline arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-d",
    "--db",
    type=str,
    help="path to HDF database",
    default=getattr(configs, "EXTRACTED_FEATURES_HDF5", "../outputs/extracted_features.hdf5")
)
ap.add_argument(
    "-m",
    "--model",
    type=str,
    help="path to output model",
    required=True
)
ap.add_argument(
    "-j",
    "--jobs",
    type=int,
    help="# of jobs to run when tuning hyperparameters",
    default=-1
)

args = vars(ap.parse_args())

# opens HDF5 database and determine the index of
# train and test split
db = h5py.File(args["db"], mode="r")
idx = int(db["labels"].shape[0] * configs.TEST_SPLIT)

# define a set of parameters we want to tune
print("[INFO]: tunning hyperparameters...")
param = {"C": [0.0001, 0.001, 0.01, 0.1, 1.0]}

# define and evaluate our model for each value of C
model = GridSearchCV(LogisticRegression(), param, n_jobs=args["jobs"])
model.fit(db["data"][idx: ], db["labels"][idx: ])
print(f"[INFO]: best hyperparameters: {model.best_params_}")

# make and display a classification report
print("[INFO]: evaluating model...")
pred = model.predict(db["data"][: idx])
print(classification_report(db["labels"][: idx], pred, target_names=db["class_labels"]))

# compute the raw accuracy of the model
accuracy = accuracy_score(db["label"][: idx], pred)
print(f"[INFO]: score: {accuracy}")

# saves the model to disk
print("[INFO]: saving model to disk...")
f = open(args["model"])
f.write(pickle.dumps(model.best_estimator_))
f.close()

# close the database
db.close()