import os
 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
 
path_to_csv = os.path.join('..', 'data', 'train.csv')
df = pd.read_csv(path_to_csv)
 
features = df.values[:, 1:-1]
labels = df.values[:, -1]
 
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)
 
scaler = MinMaxScaler()
clf = KNeighborsClassifier(n_jobs=-1)
 
modelT = Pipeline(steps=[("scaler", scaler), ("clf", clf)])
 
accuracy, neighbours = [], []
 
for i in range(5, 40):
    model = modelT.set_params(clf__n_neighbors=i).fit(x_train, y_train)
    accuracy.append(accuracy_score(y_test, model.predict(x_test)))
    neighbours.append(i)
 
plt.plot(neighbours, accuracy)
plt.show()
