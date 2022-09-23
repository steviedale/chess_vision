# %%
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

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
data = {'input_type': [], 'kernel': [], 'cross_time': [], 'cross_val_score': []}
for kernel in tqdm(('linear', 'poly_2', 'poly_3')):
    print(f"kernel: {kernel}")
    for input_label, input_data in ('images', images), ('features', features):
        print(f"\t{input_label}")

        if 'poly' in kernel:
            model = SVC(kernel='poly', C=1, random_state=0, degree=int(kernel[-1]))
        else:
            model = SVC(kernel=kernel, C=1, random_state=0)

        t0 = time.time()
        X = np.concatenate([input_data['train'], input_data['test']])
        y = np.concatenate([labels['train'], labels['test']])
        cross_val_scores = cross_val_score(model, X, y, cv=5)
        t1 = time.time()
        cross_time = (t1 - t0) / len(X)

        print(f"\tAccuracy: {cross_val_scores.mean()}")
        print(f"\tCross Time: {cross_time}")

        data['input_type'].append(input_label)
        data['kernel'].append(kernel)
        data['cross_time'].append(cross_time)
        data['cross_val_score'].append(cross_val_scores.mean())

        df = pd.DataFrame.from_dict(data)
        df.to_csv('results/cross_val_data.csv', index=False)

# %%



