from PIL import Image
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

images = os.listdir('faces')
def smaller(img):
    resized = cv2.resize(img, (0,0), fx=1/2, fy=1/2)
    return resized

plt.axis('off')

for index, image in enumerate(images):
    img = cv2.imread('faces/' + image)
    img = smaller(img)
    plt.imshow(cv2.cvtColor((img), cv2.COLOR_BGR2RGB))
    plt.show(block=False)
    category = ''
    while category not in ['m', 'f', 'o']:
        print(index, '/', len(images))
        print(image)
        category = input('Category ? ')
        
    if index % 4 == 0:
        path = 'dataset6/test'
    else:
        path = 'dataset6/train'
        
    if category == 'm':
        cv2.imwrite(path + '/male/' + image, img)
    elif category == 'f':
        cv2.imwrite(path + '/female/' + image, img)
    elif category == 'o':
        cv2.imwrite('dataset6/other/' + image, img)
        