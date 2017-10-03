# Fisherfaces classification
# Original code: https://nicholastsmith.wordpress.com/2016/02/18/eigenfaces-versus-fisherfaces-on-the-faces94-database-with-scikit-learn/

import numpy as np
from sklearn.metrics import classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from inspect import getsourcefile
from os.path import abspath
import os
import cv2

path = os.path.dirname(abspath(getsourcefile(lambda:0)))

X_train, y_train = [], []
X_test, y_test = [], []
categories = os.listdir(os.path.join(path, 'data7/test'))

for i, category in enumerate(categories):
    print(category)
    samples = os.listdir(os.path.join(path, 'data7/test', category))
    samples = [s for s in samples if 'jpg' or 'JPEG' in s]
    for sample in samples:
        img = cv2.imread(os.path.join(path, 'data7/test', category, sample), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (200, 200))
        img = img.reshape([1, 200 * 200])
        X_test.append(img)
        y_test.append(i)

for i, category in enumerate(categories):
    print(category)
    samples = os.listdir(os.path.join(path, 'data7/train', category))
    samples = [s for s in samples if 'jpg' in s]
    for sample in samples:
        img = cv2.imread(os.path.join(path, 'data7/train', category, sample), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (200, 200))
        img = img.reshape([1, 200 * 200])
        
        X_train.append(img)
        y_train.append(i)


X_train = np.reshape(X_train, (len(X_train), 40000))
X_train = np.array(X_train, dtype = np.uint8)
y_train = np.array(y_train, dtype = np.uint8)

X_test = np.reshape(X_test, (len(X_test), 40000))
X_test = np.array(X_test, dtype = np.uint8)
y_test = np.array(y_test, dtype = np.uint8)

lda = LinearDiscriminantAnalysis()
pca = PCA(n_components=(len(X_train) - 2))
pca.fit(X_train)
lda.fit(pca.transform(X_train), y_train)
y_pred = lda.predict(pca.transform(X_test))
print(classification_report(y_test, y_pred))
