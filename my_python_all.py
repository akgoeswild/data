#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:57:34 2019

@author: vipss
"""
# =============================================================================
# #K-means for iris
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

iris=load_iris()
X=iris.data
y=iris.target
y_names=iris.target_names

#elbow curve for number of clusters
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++', random_state=64)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

sc=StandardScaler()
X_scaled=sc.fit_transform(X)

#elbow curve for number of clusters
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++', random_state=64)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion(Cost Function)')
plt.show()

kmeans = KMeans(n_clusters=3, init='k-means++', random_state=64)
kmeans.fit(X)
clust_labels = kmeans.labels_
cent = kmeans.cluster_centers_
cm=confusion_matrix(y, kmeans.labels_)

plt.scatter(X[:, 0], X[:, 1], c=clust_labels, s=50, cmap='viridis')
plt.scatter(cent[:, 0], cent[:, 1], c='black', s=200, alpha=0.5);


# =============================================================================
# #SVM for wine
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

wine2=load_wine()
X=wine2.data
y=wine2.target

dataset=pd.DataFrame(np.column_stack((y,X)), columns=['Class', 'Alcohol', 'Malic Acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Tot_phenols', 'Flavanoids', 'Non_flavanoid_phenols', 'Proanthocyanins', 'Colour', 'Hue', 'OD280/OD315', 'Proline'])

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=64)
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

model=SVC( kernel='rbf')
model.fit(X_train, y_train)
y_pred=model.predict(X_test)

cm=confusion_matrix(y_test, y_pred)
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Precision: ', precision_score(y_test, y_pred, average="weighted"))
print('Recall: ', recall_score(y_test, y_pred, average="weighted"))


# =============================================================================
# svm for boston
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.datasets import load_boston

boston=load_boston()
dataset=pd.DataFrame(np.column_stack((boston.data, boston.target)), columns=np.append(boston.feature_names,'ans'))
X=boston.data
y=boston.target

print(pd.DataFrame(y).describe())
descr=pd.DataFrame(y).describe().iloc[:,0].values.tolist()

for i in range(len(y)):
    if y[i]<=descr[4]:
        y[i]="1"
    elif y[i]>descr[4] and y[i]<=descr[5]:
        y[i]="2"
    elif y[i]>descr[5] and y[i]<=descr[6]:
        y[i]="3"
    else:
        y[i]="4"
        
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=64)
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


model=SVC( kernel='rbf')
model.fit(X_train, y_train)
y_pred=model.predict(X_test)

cm=confusion_matrix(y_test, y_pred)
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Precision: ', precision_score(y_test, y_pred, average="weighted"))
print('Recall: ', recall_score(y_test, y_pred, average="weighted"))


# =============================================================================
# svm for diabetes
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.datasets import load_diabetes

diabetes=load_diabetes()
dataset=pd.DataFrame(np.column_stack((diabetes.data, diabetes.target)), columns=np.append(diabetes.feature_names, 'answer'))
X=diabetes.data
y=diabetes.target

abc=pd.DataFrame(y).describe()
descr=abc.iloc[:, 0].values.tolist()

for i in range(len(y)):
    if y[i]<=descr[4]:
        y[i]="1"
    elif y[i]>descr[4] and y[i]<=descr[5]:
        y[i]="2"
    elif y[i]>descr[5] and y[i]<=descr[6]:
        y[i]="3"
    else:
        y[i]="4"
        
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=64)
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


model=SVC( kernel='linear')
model.fit(X_train, y_train)
y_pred=model.predict(X_test)

cm=confusion_matrix(y_test, y_pred)
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Precision: ', precision_score(y_test, y_pred, average="weighted"))
print('Recall: ', recall_score(y_test, y_pred, average="weighted"))


# =============================================================================
# visualize neural network
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential #Sequential Models
from keras.layers import Dense #Dense Fully Connected Layer Type
from sklearn.preprocessing import StandardScaler
from ann_visualizer.visualize import ann_viz;
from sklearn.datasets import load_wine

wine=load_wine()
X=wine.data
y=wine.target


sc=StandardScaler()
X_scaled=sc.fit_transform(X)

b=LabelBinarizer()
y=b.fit_transform(y)

model=Sequential()
model.add(Dense(16, input_shape=(X.shape[1], ), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X_scaled, y, epochs=100, batch_size=10)

ann_viz(model, title='Neural Network', filename='vipss')