import numpy as np
import cv2
import random


# Generator class
# Inspired from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html

class Generator():
    def __init__(self, width = 60, height = 90, channels = 1, batch_size = 64):
        self.width = width
        self.height = height
        self.channels = channels
        self.batch_size = batch_size
    
    def to_categorical(self, y, n):
        return np.array([[1 if y[i] == j else 0 for j in range(n)]
                     for i in range(y.shape[0])])
        
    def generate_data(self, features, labels):     
        batch_features = []
        batch_labels = np.zeros((self.batch_size,1))
        n = len(set(labels))
        for i, feature in enumerate(features):
            temp = cv2.imread(feature)
            temp = cv2.resize(temp, (self.width, self.height))
            batch_features.append(temp)
            batch_labels[i] = labels[i]
        batch_features = np.reshape(batch_features, (len(batch_features), self.height, self.width, self.channels))
        
        if n == 2:
            return batch_features, batch_labels
        return batch_features, self.to_categorical(batch_labels, n)

    def generate(self, features, labels, shuffle  = True, return_labels = True):
        while True:
            combined = list(zip(features, labels))
            if shuffle:
                random.shuffle(combined)
            arranged_features, arranged_labels = zip(*combined)
            
            iterations = int(len(arranged_features) / self.batch_size)
            for i in range(iterations):
                X, y = self.generate_data(arranged_features[i*self.batch_size:(i+1)*self.batch_size], arranged_labels[i*self.batch_size:(i+1)*self.batch_size])
                if return_labels:
                    yield X, y
                else:
                    yield X