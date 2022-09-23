# %%
# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.ensemble import AdaBoostClassifier

import cv2
from tqdm import tqdm
import imutils
import numpy as np
import time

# %%
df_dict = {
    'train': pd.read_csv('/home/stevie/datasets/chess_vision/256x256/dataframes/train.csv'),
    'test': pd.read_csv('/home/stevie/datasets/chess_vision/256x256/dataframes/test.csv'),
}

# %%
# grab the list of images that we'll be describing
print("[INFO] describing images...")
# initialize the raw pixel intensities matrix, the features matrix,
# and labels list
images = {}
features = {}
labels = {}
for set_str in 'train', 'test':
    images[set_str] = []
    features[set_str] = []
    labels[set_str] = []

# %%
def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()

# %%
def extract_color_histogram(image, bins=(8, 8, 8)):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])
	# handle normalizing the histogram if we are using OpenCV 2.4.X
	if imutils.is_cv2():
		hist = cv2.normalize(hist)
	# otherwise, perform "in place" normalization in OpenCV 3 (I
	# personally hate the way this is done
	else:
		cv2.normalize(hist, hist)
	# return the flattened histogram as the feature vector
	return hist.flatten()

# %%
# loop over the input images
for set_str, df in df_dict.items():
	print(len(df))
	for i, row in tqdm(df.iterrows()):
		# load the image and extract the class label (assuming that our
		# path as the format: /path/to/dataset/{class}.{image_num}.jpg
		image = cv2.imread(row['path'])
		label = row['label']
		# extract raw pixel intensity "features", followed by a color
		# histogram to characterize the color distribution of the pixels
		# in the image
		pixels = image_to_feature_vector(image)
		hist = extract_color_histogram(image)
		# update the raw images, features, and labels matricies,
		# respectively
		images[set_str].append(pixels)
		features[set_str].append(hist)
		labels[set_str].append(label)

# %%
for set_str in 'train', 'test':
    images[set_str] = np.array(images[set_str])
    features[set_str] = np.array(features[set_str])
    labels[set_str] = np.array(labels[set_str])

    print(set_str)
    print("[INFO] pixels matrix: {:.2f}MB".format(images[set_str].nbytes / (1024 * 1000.0)))
    print("[INFO] features matrix: {:.2f}MB".format(features[set_str].nbytes / (1024 * 1000.0)))

# %%
len(images['train'])

# %%
data = {'input_type': [], 'n': [], 'fit_time': [], 'eval_time': [], 'accuracy': []}
for n in tqdm((1, 2, 3, 4, 5, 10, 15, 20)):
    print(f"n: {n}")
    for input_label, input_data in ('images', images), ('features', features):
        print(f"\t{input_label}")

        model = AdaBoostClassifier(n_estimators=n, random_state=0)

        t0 = time.perf_counter()
        model = model.fit(input_data['train'], labels['train'])
        t1 = time.perf_counter()
        fit_time = (t1 - t0) / len(input_data['train'])

        #Predict the response for test dataset
        t0 = time.perf_counter()
        y_pred = model.predict(input_data['test'])
        t1 = time.perf_counter()
        eval_time = (t1 - t0) / len(input_data['test'])

        # Model Accuracy, how often is the classifier correct?
        acc = metrics.accuracy_score(labels['test'], y_pred)

        print(f"\tAccuracy: {acc}")
        print(f"\tFit Time: {fit_time}")
        print(f"\tEval Time: {eval_time}")

        data['input_type'].append(input_label)
        data['n'].append(n)
        data['eval_time'].append(eval_time)
        data['fit_time'].append(fit_time)
        data['accuracy'].append(acc)

        df = pd.DataFrame.from_dict(data)
        df.to_csv('results/data.csv', index=False)
