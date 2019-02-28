from HoG import getGradientElement, extractHistograms, extractHOGFromHistograms, extractHOGFromHistograms, getHOGDescriptor
from SVM import SVM_init
from sklearn.externals import joblib
import numpy as np
import os
import cv2
from setting import BASE_DIR, GRAY_PATH;

#path = os.path.join(GRAY_PATH, "Dahlia","2dah035.jpg")
#img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

#result, angle, resultX, resultY = getGradientElement(img, 180)
#cv2.imshow("Gx", resultX)
#cv2.imshow("Gy", resultY)
#cv2.imshow("G", result)
#cv2.waitKey(0)


label = {
    "Cal": 0,
    "Hor": 1,
    "Lot": 2,
}

nameFromLabel = {
    0: 'Calla lily',
    1: 'Hortensia',
    2: 'Lotus'
}

NAME_LIST = {
    "Cal":'cal',
    "Hor":'hor',
    "Lot":'lot'
}

svm = SVM_init(100,0.01,'rbf')

def list_item(input_path):
    for item in os.listdir(input_path):
        if item in NAME_LIST:
            path = os.path.join(input_path, item)
            for hoa_item in os.listdir(path):
               if hoa_item.endswith('.png') or hoa_item.endswith('.jpg'):
                    path_hoa = os.path.join(path, hoa_item)
                    yield path_hoa                               


input_path = "E:/Train"
fileName = os.path.join(input_path, "train.pkl")

#def getDescription(input_path):
#    X= []
#    y=[]
#    for index, item in enumerate(list_item(input_path)):
#        name = os.path.split(os.path.dirname(item))[1]
#        img = cv2.imread(item, cv2.IMREAD_GRAYSCALE)
#        hog = getHOGDescriptor(img, 10, 20, 10, 'L2', 9, 180)
#        X.append(hog)
#        y.append(label[name])
#    return X, y

#X, y = getDescription(input_path)
#svm.fit(X,y)
#joblib.dump(svm, fileName)

test_path  = "E:/Test"
fileTest = os.path.join(test_path,"2lot163.png")

img_test = cv2.imread(fileTest, cv2.IMREAD_GRAYSCALE)
hog = getHOGDescriptor(img_test, 10, 20, 10, 'L2', 9, 180)
hog = hog.reshape(1,-1)
svm = joblib.load(fileName)
predict = svm.predict(hog)
predict = list(predict.flatten())

#print(predict[0])
print(nameFromLabel[predict[0]])
