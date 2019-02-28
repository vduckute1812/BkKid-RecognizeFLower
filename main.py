from __future__ import print_function

import cv2
import math
import os
import time
import numpy as np
import pickle
from sklearn.externals import joblib
from setting import OUT_PATH, BINARY_PATH, GRAY_PATH, BASE_DIR, NAME_LIST, GRAY_HU, OUT_TRUE_FILE_CLUSION, OUT_FALSE_FILE_CLUSION, ORIGIN_PATH
from tempfile import mkdtemp
from helper import huMonents, logarit, list_item,nameGrToEgde, resize, kmeans
from tempfile import TemporaryFile
from xlwt import Workbook,easyxf,Style
from xlrd import open_workbook
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from xlutils.copy import copy
from HoG import get_hog, extractHistograms, extractHOGFromHistograms, descriptionHoG, getGradientElement
from SVM import SVM_init, estimateParam, createExcelFile, resultSVM, descriptionHu, decreaseNumFlower




#----------
size=len(NAME_LIST)

folder=str(size)+"bong"
folder=os.path.join(BASE_DIR,"ChiTietHoaHog",folder,"TrueClusion")
false_folder=os.path.join(BASE_DIR,"ChiTietHoaHog",folder,"FalseClusion")
outTrueFile=OUT_TRUE_FILE_CLUSION
outFalseFile=OUT_FALSE_FILE_CLUSION

svm =SVM_init(100,0.01,'rbf')

for num_test in range(100, 201, 50):
    path = os.path.join(BASE_DIR, "HuImage", str(num_test))
    fileName = str(size)+"bong"+str(num_test)+'.xls'
    createExcelFile(fileName)
    hog = get_hog()
    X, y, num_flower, pathList, timePath = descriptionHoG(path, hog, num_test)
    resultSVM(X, y, num_flower, pathList, svm,fileName,timePath, OUT_TRUE_FILE_CLUSION, OUT_FALSE_FILE_CLUSION, num_test)

#size = len(NAME_LIST)
#num = 200
#fileName = "Hog"+str(size)+str(num)+".xls"
#createExcelFile(fileName)
#hog = get_hog()
#X, y, num, path = description(GRAY_PATH, hog)
#print(num)
#estimateParam(X,y)
#print(num)
#svm =SVM_init(100,0.01,'rbf')
#resultSVM(X,y,num,svm,fileName)
#estimateParam(X,y)
 
#histograms = extractHistograms(img, 10, 9)
#print(histograms.shape) 
#blocks = extractHOGFromHistograms(histograms, 2, 1, 'L2')
#HoG = np.array(blocks).flatten()
#HoG = HoG.tolist()
#print(HoG.shape)

#for index in arr:
#    print(index)




cv2.waitKey(0)
##----------
# size = len(NAME_LIST)
# num=100
# fileName = str(size)+str(num)+".xls"

# folder = str(size)+"bong"
# folder = os.path.join(BASE_DIR, "ChiTietHoa",folder,"TrueClusion")
# false_folder = os.path.join(BASE_DIR, "ChiTietHoa",folder,"FalseClusion")
# outTrueFile = OUT_TRUE_FILE_CLUSION
# outFalseFile = OUT_FALSE_FILE_CLUSION
# createExcelFile(fileName)
# print(folder)
# X, y, num, path = description(GRAY_PATH)
# print(X)
# print(y)
# print(num)
# X,y, num_flower, num_test, pathList = decreaseNumFlower(GRAY_PATH,false_folder, num)

# print(num_test)
# print(UPDATE_PATH)
# print(num_flower)
# svm =SVM_init(1e5,1e-4,'rbf')
# detailResult(X,y,num_test,pathList,svm,fileName,outTrueFile,outFalseFile)


###--------
#app = SVM_OptimalParameter(X,y)
#app.adjust_SVM()
#svm = cv2.ml.SVM_create()
#svm.setGamma(0.03)
#svm.setC(15)
#svm.setKernel(cv2.ml.SVM_RBF)
#svm.setType(cv2.ml.SVM_C_SVC)
#svm.train(X, cv2.ml.ROW_SAMPLE,y)
#svm.save('data.dat')
#svm = cv2.ml.SVM_load('data.dat')
#path = "C:/Users/DUC13T3/Documents/Visual Studio 2013/Projects/FlowerDetect-Core/SVMupdate/SVMupdate/Img/3cal038.jpg"
#img = cv2.imread(path)
#logHu = logarit(img)
#print(logHu)

#trainData = np.float32(logHu).reshape(-1,7)
#result = svm.predict(trainData)[1].ravel()
#print(result)
##path = os.path.join(GRAY_PATH,"Dahlia","3dah001.jpg")
##img = cv2.imread(path)
##cv2.imshow("bla",img)
##loghu = logarit(img)
##print(loghu)
##path ="C:/Users/DUC13T3/Documents/Visual Studio 2013/Projects/FlowerDetect-Core/hoarum.jpg"
##img= cv2.imread(path)
##cv2.imshow("bla",img)
##oriToScale(path)
# cv2.waitKey(0)