import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
from string import ascii_letters
import os
import mymodule
import pickle
import cv2


path_to_model = os.path.join('..', 'data', 'model.pickle')
with open(path_to_model, 'rb') as f:
	model = pickle.load(f)

test_cat_path = os.path.join('..', 'data', 'test')
for image_name in os.listdir(test_cat_path):
	path = os.path.join(test_cat_path, image_name)
	image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
	
	if len(image.shape) > 2 and image.shape[2] == 4:
		image = image[:,:,3]
	else:
		if len(image.shape) == 3:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);
		(_, image) = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
	
	image = mymodule.clip_image(image)
	features = np.array(0);
	features = mymodule.calc_features(image, features)
	
	
	y = model.predict(features)
	y = ascii_letters[model.predict(features)[0]]
	
	cv2.imshow(y + ' ' + path, image)

cv2.waitKey(0)
cv2.destroyAllWindows()
