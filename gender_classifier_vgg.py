import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras_vggface.vggface import VGGFace
from inspect import getsourcefile
from os.path import abspath
from generator import Generator

path = os.path.dirname(abspath(getsourcefile(lambda:0)))

categories = os.listdir(os.path.join(path, 'data7/test'))
y_train = []
X_train = []
y_test = []
X_test = []
batch_size = 10
print('Getting samples')

for i, category in enumerate(categories):
    print(category)
    samples = os.listdir(os.path.join(path, 'data7/test', category))
    for sample in samples:
        X_test.append(os.path.join(path, 'data7/test', category, sample))
        y_test.append(i)

for i, category in enumerate(categories):
    print(category)
    samples = os.listdir(os.path.join(path, 'data7/train', category))
    for sample in samples:
        X_train.append(os.path.join(path, 'data7/train', category, sample))
        y_train.append(i)


training_generator = Generator(width = 224, height = 224, channels = 3, batch_size = batch_size).generate(X_train, y_train, return_labels = False, shuffle = False)
testing_generator = Generator(width = 224, height = 224, channels = 3, batch_size = batch_size).generate(X_test, y_test, return_labels = False, shuffle = False)

def save_features():
    model = VGGFace(include_top=False, input_shape=(224, 224, 3), weights='vggface', pooling = 'avg')              
    bottleneck_features_train = model.predict_generator(training_generator, len(X_train) // batch_size)
    np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)
    
    bottleneck_features_test = model.predict_generator(testing_generator, len(X_test) // batch_size)
    np.save(open('bottleneck_features_testing.npy', 'wb'), bottleneck_features_test)


def create_model():
    train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
    train_labels = np.array([0] * 80 + [1] * 140)
    train_labels = train_labels[:len(train_labels) - (len(train_labels) % batch_size)]
    
    validation_data = np.load(open('bottleneck_features_testing.npy', 'rb'))
    validation_labels = np.array([0] * 30 + [1] * 40)
    validation_labels = validation_labels[:len(validation_labels) - (len(validation_labels) % batch_size)]
    
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=train_data.shape[1:]))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(train_data, train_labels,
              epochs=100,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))

    score = model.evaluate(validation_data, validation_labels)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print("Baseline Error: %.2f%%" % (100-score[1]*100))
