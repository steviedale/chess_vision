# %%
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
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
images = {}
features = {}
labels = {}
for set_str in 'train', 'test':
    images[set_str] = []
    features[set_str] = []
    labels[set_str] = []

# %%
def image_to_feature_vector(image, size=(32, 32)):
	return cv2.resize(image, size).flatten()

# %%
def extract_color_histogram(image, bins=(8, 8, 8)):
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])
	if imutils.is_cv2():
		hist = cv2.normalize(hist)
	else:
		cv2.normalize(hist, hist)
	return hist.flatten()

# %%
for set_str, df in df_dict.items():
	print(len(df))
	for i, row in tqdm(df.iterrows()):
		image = cv2.imread(row['path'])
		label = row['label']
		pixels = image_to_feature_vector(image)
		hist = extract_color_histogram(image)
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
data = {'input_type': [], 'criterion': [], 'cross_time': [], 'cross_val_score': []}
for criterion in tqdm(('entropy', 'gini', 'log_loss')):
    print(f"criterion: {criterion}")
    for input_label, input_data in ('images', images), ('features', features):
        model = DecisionTreeClassifier(criterion=criterion)

        t0 = time.time()
        X = np.concatenate([input_data['train'], input_data['test']])
        y = np.concatenate([labels['train'], labels['test']])
        scores = cross_val_score(model, X, y, cv=5)
        t1 = time.time()
        eval_time = (t1 - t0) / len(X)

        print(f"\tAccuracy: {scores.mean()}")
        print(f"\tEval Time: {eval_time}")

        data['input_type'].append(input_label)
        data['criterion'].append(criterion)
        data['cross_time'].append(eval_time)
        data['cross_val_score'].append(scores.mean())

        df = pd.DataFrame.from_dict(data)
        df.to_csv('results/cross_val_data.csv', index=False)

# %%



