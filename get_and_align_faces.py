from imutils.face_utils import FaceAligner
from dlib import rectangle
import face_recognition
import imutils
import dlib
import cv2
import os

def resize(img):
    resized = cv2.resize(img, (0,0), fx=2.5, fy=2.5)
    return resized

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:\\WinPython-64bit-3.5.3.0Qt5\\python-3.5.3.amd64\\Lib\\site-packages\\face_recognition_models\\models\\shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=224)

images = os.listdir('Posters')
for i, index in enumerate(images):
    print(index, ':', i, '/', len(images))
    image_path = 'F:\\Posters\\Posters\\' + index + '\\' + index + '.jpg'
    img = cv2.imread(image_path)
    img = resize(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_locations = face_recognition.face_locations(img, number_of_times_to_upsample=0)
    
    for j, face_location in enumerate(face_locations):
        
        top, right, bottom, left = face_location
        dlib_rect = rectangle(left, top, right, bottom) #convert to dlib rect object
        faceOrig = imutils.resize(img[top:bottom, left:right], width=224)
        faceAligned = fa.align(img, gray, dlib_rect)        
        cv2.imwrite('faces/' + index + '_' + str(j) + '.jpg', faceAligned)
