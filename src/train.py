import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import os
import pickle
import mymodule

mymodule.prepare_csv()

path_to_csv = os.path.join('..', 'data', 'train.csv')
df = pd.read_csv(path_to_csv)
y = df.target.values.ravel().astype(int)
X = df.drop(['target'], axis = 1)

scaler = MinMaxScaler()
clf = KNeighborsClassifier(n_neighbors=10, weights='distance')
model = Pipeline(steps=[("scaler", scaler), ("clf", clf)])

model.fit(X, y)

path_to_model = os.path.join('..', 'data', 'model.pickle')
with open(path_to_model, 'wb') as f:
    pickle.dump(model, f)
print("Writing model to", path_to_model)
