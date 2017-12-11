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

path_to_csv = os.path.join('..', 'data', 'train.csv')
df = pd.read_csv(path_to_csv)
y = df.target.values.ravel().astype(int)
#~ X = df.drop('target', axis = 1)
X = df.drop(['target'], axis = 1)

scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

clf = KNeighborsClassifier(n_neighbors=5)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
print(scores)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))

df_test = pd.DataFrame(y_test)
df_pred = pd.DataFrame(y_pred)
df_test = pd.get_dummies(df_test[0])
df_pred = pd.get_dummies(df_pred[0])

#~ df_pred.columns = list(ascii_letters[:26])
#~ df_test.columns = list(ascii_letters[:26])
#~ corr = df_test.corrwith(df_pred)


def percConvert(ser):
  return ser/float(ser[-1])

df_y = pd.DataFrame({'true':y_test, 'pred':y_pred}) 
ct = pd.crosstab(df_y["true"],df_y["pred"],margins=True).apply(percConvert, axis=1)
sns.heatmap(ct, cmap='Greens')
plt.show()



