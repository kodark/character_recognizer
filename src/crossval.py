import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
from string import ascii_letters
import os
import mymodule


mymodule.prepare_csv()

path_to_csv = os.path.join('..', 'data', 'train.csv')
df = pd.read_csv(path_to_csv)
 
features = df.values[:, :-1]
labels = df.values[:, -1]
 
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)
 
scaler = MinMaxScaler()
clf = KNeighborsClassifier(weights='distance', n_jobs=-1)
 
modelT = Pipeline(steps=[("scaler", scaler), ("clf", clf)])
 
accuracy, neighbours, stds = [], [], []
 
for i in range(5, 40):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    model = modelT.set_params(clf__n_neighbors=i).fit(x_train, y_train)
    scores = cross_val_score(model, features, labels, cv=cv, scoring='accuracy')
    accuracy.append(np.mean(scores))
    stds.append(np.std(scores))
    neighbours.append(i)

stds = np.array(stds)
accuracy = np.array(accuracy)

i = np.argmax(accuracy)
print(neighbours[i], accuracy[i])
plt.plot(neighbours, accuracy)
plt.title('Веса соседей обратно пропорциональны расстоянию')
plt.fill_between(neighbours, accuracy - 1.96*stds,
                 accuracy + 1.96*stds, alpha=0.2)
plt.show()

