from __future__ import print_function

import math
import time
import os
import cv2
import numpy as np
import pickle
from sklearn.externals import joblib
from setting import OUT_PATH, BINARY_PATH, GRAY_PATH, BASE_DIR, NAME_LIST, GRAY_HU
from tempfile import mkdtemp
from helper import huMonents, logarit, list_item,nameGrToEgde
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

label = {
    "Dahlia": 0,
    "Ixora": 1,
     "Lily": 2,
    "Rose": 3,
    "Sun": 4,
    "Hoa But": 5,
    "Hoa Cam Tu Cau":6,
    "Hoa Canh Buom":7,
    "Hoa Cuc Trang":8,
    "Hoa Hong Mon":9,
    "Hoa Mao Ga":10,
    "Hoa Rum":11,
    "Hoa Sen":12,
    "Hoa Thien Dieu":13,
    "Hoa Van Tho":14
}
get_name = {
    0: "Dahlia",
    1: "Ixora",
    2: "Lily",
    3: "Rose",
    4: "Sun",
    5: "Hoa But",
    6: "Hoa Cam Tu Cau",
    7: "Hoa Canh Buom",
    8: "Hoa Cuc Trang",
    9: "Hoa Hong Mon",
    10:"Hoa Mao Ga",
    11:"Hoa Rum",
    12:"Hoa Sen",
    13:"Hoa Thien Dieu",
    14:"Hoa Van Tho"
}

def huGrayMixEdge(PATH_GRAY, PATH_EDGE):
    X=[]
    y=[]
    tmp=[]
    num_of_flower = {
        "Dahlia": 0,
        "Ixora": 0,
        "Lily": 0,
        "Rose": 0,
        "Sun": 0,
        "Hoa But": 0,
        "Hoa Cam Tu Cau": 0,
        "Hoa Canh Buom": 0,
        "Hoa Cuc Trang": 0,
        "Hoa Hong Mon": 0,
        "Hoa Mao Ga": 0,
        "Hoa Rum": 0,
        "Hoa Sen": 0,
        "Hoa Thien Dieu": 0,
        "Hoa Van Tho": 0
    }
    for index,item_path in enumerate(list_item(PATH_GRAY)):
        del tmp[:]
        name = os.path.split(os.path.dirname(item_path))[1]
        print(name)
        dir2 = os.path.basename(item_path)
        pathImageEdge = os.path.join(PATH_EDGE, name, dir2)
        image = cv2.imread(item_path)
        image2 = cv2.imread(pathImageEdge,cv2.IMREAD_GRAYSCALE)
        data2 =list(np.float32(image2).flatten())
        hu = logarit(image)
        #for x in hu:
        #    tmp.append(x)
        for x2 in data2:
            if(x2>200):
                tmp.append(255)
            else:
                tmp.append(0)
        num_of_flower[name] += 1

        X.append(tmp)
        y.append(label[name])
        #print(X)
    return X,y,num_of_flower

def huGrayMixHuEdge(PATH_GRAY, PATH_EDGE):
    X=[]
    y=[]
    tmp=[]
    num_of_flower = {
        "Dahlia": 0,
        "Ixora": 0,
        "Lily": 0,
        "Rose": 0,
        "Sun": 0,
        "Hoa But": 0,
        "Hoa Cam Tu Cau": 0,
        "Hoa Canh Buom": 0,
        "Hoa Cuc Trang": 0,
        "Hoa Hong Mon": 0,
        "Hoa Mao Ga": 0,
        "Hoa Rum": 0,
        "Hoa Sen": 0,
        "Hoa Thien Dieu": 0,
        "Hoa Van Tho": 0
    }
    for index,item_path in enumerate(list_item(PATH_GRAY)):
        del tmp[:]
        name = os.path.split(os.path.dirname(item_path))[1]
        print(name)
        dir2 = os.path.basename(item_path)
        pathImageEdge = os.path.join(PATH_EDGE, name, dir2)
        image = cv2.imread(item_path)
        image2 = cv2.imread(pathImageEdge)
        huGray = logarit(image)
        huEdge = logarit(image2)
        for i in huGray:
            tmp.append(i)
        for j in huEdge:
            tmp.append(j)
        num_of_flower[name] += 1
        X.append(tmp)
        y.append(label[name])
        #print(X)
    return X,y,num_of_flower

def descriptionHu(PATH_FLOWER):
    num_of_flower = {
        "Dahlia": 0,
        "Ixora": 0,
        "Lily": 0,
        "Rose": 0,
        "Sun": 0,
        "Hoa But": 0,
        "Hoa Cam Tu Cau": 0,
        "Hoa Canh Buom": 0,
        "Hoa Cuc Trang": 0,
        "Hoa Hong Mon": 0,
        "Hoa Mao Ga": 0,
        "Hoa Rum": 0,
        "Hoa Sen": 0,
        "Hoa Thien Dieu": 0,
        "Hoa Van Tho": 0
    }
    X = []
    y = []
    pathFlower = []
    for index,item_path in enumerate(list_item(PATH_FLOWER)):
        name = os.path.split(os.path.dirname(item_path))[1]
        image = cv2.imread(item_path)
        #hu = huMonents(image)
        hu = logarit(image)
        #image = image.flatten()
        num_of_flower[name] += 1
        X.append(hu)
        y.append(label[name])
        pathFlower.append(item_path)
    return X, y, num_of_flower, pathFlower

def SVM_init(C, gamma, kernel):
    svm = SVC(C=C, kernel=kernel, gamma=gamma)
    return svm

def decreaseNumFlower(PATH_TRUE_FLOWER, PATH_FALSE_FLOWER, num_test):
    num_of_flower = {
        "Dahlia": 0,
        "Ixora": 0,
        "Lily": 0,
        "Rose": 0,
        "Sun": 0,
        "Hoa But": 0,
        "Hoa Cam Tu Cau": 0,
        "Hoa Canh Buom": 0,
        "Hoa Cuc Trang": 0,
        "Hoa Hong Mon": 0,
        "Hoa Mao Ga": 0,
        "Hoa Rum": 0,
        "Hoa Sen": 0,
        "Hoa Thien Dieu": 0,
        "Hoa Van Tho": 0
    }
    test = {
        "Dahlia": 0,
        "Ixora": 0,
        "Lily": 0,
        "Rose": 0,
        "Sun": 0,
        "Hoa But": 0,
        "Hoa Cam Tu Cau": 0,
        "Hoa Canh Buom": 0,
        "Hoa Cuc Trang": 0,
        "Hoa Hong Mon": 0,
        "Hoa Mao Ga": 0,
        "Hoa Rum": 0,
        "Hoa Sen": 0,
        "Hoa Thien Dieu": 0,
        "Hoa Van Tho": 0
    }

    X = []
    y = []
    pathList = []
    timePath = []
    for index,item_path in enumerate(list_item(PATH_TRUE_FLOWER)):
        t = time.clock()
        name = os.path.split(os.path.dirname(item_path))[1]
        falseFolderFlower = os.path.join(PATH_FALSE_FLOWER, name)
        DIR = os.path.join(PATH_TRUE_FLOWER, name)
        numOfImage = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
        image = cv2.imread(item_path)
        #hu = huMonents(image)
        t = time.clock()
        hu = logarit(image)
        timeProcessing = time.clock()-t
        #image = image.flatten()
        num_of_flower[name] += 1
         
        if(num_of_flower[name]>num_test):
            continue
        X.append(hu)
        y.append(label[name])
        pathList.append(item_path)
        test[name]+=1
        timePath.append(timeProcessing)
	#Add Hu feature into list
        #if(numOfImage < num_test and num_of_flower[name]==numOfImage):
        #    for hoa_item in os.listdir(falseFolderFlower):
        #        name = os.path.split(os.path.dirname(hoa_item))[1]
        #        image2 = cv2.imread(hoa_item)
        #        hu2 = logarit(image2)
        #        X.append(hu2)
        #        y.append(label[name])
        #        test[name]+=1
        #        numOfImage+=1
        #        pathList.append(hoa_item)
        #        if(numOfImage < num_test):
        #            break                        
    return X, y, num_of_flower, test, pathList, timePath

def resultSVM(X, y, num_of_flower, pathList,svm, nameFile, timeExecute, outTrueFile, outFalseFile, num_test):
    percentData = []
    item = 0
    item2 = 0
    num=0
    result = []
    table = []
    table2 = []
    nameFlowerWrite = []
    
    ResultDirectory = str(num_test)
    size = len(NAME_LIST)

    for x in range(size):
        detectTable = [0]*len(label)
        nameFlower = get_name[y[num]]
        start = num
        num += num_of_flower[nameFlower]
        listPathFlower = pathList[start:num]
        #num+=200
        print(">>>>Flower : ", nameFlower)
        directory = os.path.join("HogGrey", str(num_test), nameFlower)
        if not os.path.exists(directory):
            os.makedirs(directory)    
        print(">>>>Flower : ", nameFlower)
        del result[:]
        del table [:]
        del table2 [:]
        for index in range(start, num, int(num_of_flower[nameFlower]/10)):
            fileName = os.path.join(directory,str(index)+".pkl")
            s = index
            f = index+int(num_of_flower[nameFlower]/10) if index+int(num_of_flower[nameFlower]/10)<num else num
            inTest = X[s:f]
    #-------Training process-----------------------
            inTrain = X[:s] + X[f:]
            labelTrain = y[:s] + y[f:]
            svm.fit(inTrain, labelTrain) 
            joblib.dump(svm, fileName)
            print(index)

    #------Detect process after training----------

    #        svm = joblib.load(fileName)
    #        predict = svm.predict(inTest)
    #        result.append(list(predict.flatten()))
    #        list_value = calculate(list((predict.flatten())), 15)        
    #        table.append(list_value)
    #        for lb_result in predict:
    #            detectTable[lb_result]+=1
    #            table2.append(lb_result)
        
    #    detectTable = map(lambda x:  x*100/num_of_flower[nameFlower],detectTable)
    #    percentData.append(list(detectTable))
    #    nameFlowerWrite.append(nameFlower)
        
    #    saveFile(table, nameFlower, num_of_flower[nameFlower], item, 0, nameFile)    
    #    item += len(table)+5
    #    table2=[table2]
    #    saveFile2(table2, nameFlower, num_of_flower[nameFlower],listPathFlower, item2, 1, nameFile, outTrueFile,outFalseFile, timeExecute[start:num])
    #    item2 += len(table2) + 5
    #print("percent detail")   
    #saveFile3(percentData,nameFlowerWrite,2, nameFile)
    #print(percentData)
    #------------------------------------------

def detailResult(X,y,num_of_flower, pathList,svm, nameFile, outTrueFile, outFalseFile):
    percentData = []
    nameFlowerWrite = []
    size = len(NAME_LIST)
    #ResultDirectory = "trainOfTrue"+str(size)
    ResultDirectory = "HuGrey"+str(size)    
    #ResultDirectory = "trainOfTrue"+str(size)
    item2 = 0
    num=0
    result2 = []
    table2 = []
    for x in range(size):
        nameFlower = get_name[y[num]]
        start = num
        num += num_of_flower[nameFlower]
        listPathFlower = pathList[start:num]
        print(">>>>Flower : ", nameFlower)
        #directory = os.path.join(BASE_DIR, ResultDirectory, nameFlower)
        directory = os.path.join(GRAY_HU, ResultDirectory, nameFlower)
        if not os.path.exists(directory):
            os.makedirs(directory)    
         #reset data
        del result2[:]
        del table2 [:]
        detectTable = [0]*len(label)

        for index in range(start, num):
            fileName = os.path.join(directory,str(index)+".pkl")
            inTest = X[index]
            inTrain = X[:index] + X[index+1:]
            labelTrain = y[:index] + y[index+1:]
            svm.fit(inTrain, labelTrain)
            joblib.dump(svm, fileName)
            print(index)
            svm = joblib.load(fileName)
            predict = svm.predict([inTest])
            print(predict)
            result2.append(predict)
            detectTable[predict[0]] += 1
        table2.append(result2)
        detectTable = map(lambda x:  x*100/num_of_flower[nameFlower],detectTable)
        percentData.append(list(detectTable))
        nameFlowerWrite.append(nameFlower)
        #print(table2)
        saveFile2(table2, nameFlower, num_of_flower[nameFlower],listPathFlower, item2, 1, nameFile, outTrueFile,outFalseFile)
        item2 += len(table2) + 4
    print("percent detail")   
    saveFile3(percentData,nameFlowerWrite,2, nameFile)
    print(percentData)


def estimateParam(X, y):
    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-5,1e-5,1e-4,1e-3,1e-2,1e-1],
                         'C': [100,1000,10000,1e5,1e6]}]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                           scoring='%s_macro' % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

def estimateParam2(X,y):
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(X, y)

    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))


def calculate(list_value, item):
    buckets = [0]*item
    for i in (list_value):
        buckets[i]+=1
    return buckets

def createExcelFile(nameFile):
	w = Workbook()
	ws = w.add_sheet('SVM')
	ws2 = w.add_sheet('Detail')
	ws3 = w.add_sheet('percentDetail')
	w.save(nameFile)
	w.save(TemporaryFile())


def saveFile(data, nameFlower, numFlower, item, index, nameFile):

    style = easyxf(
    'pattern: pattern solid, fore_colour red;'
    'align: vertical center, horizontal center;'
    )
    style2 = easyxf(
    'align: vertical center, horizontal center;'
    )

    rb = open_workbook(nameFile,formatting_info=True)
    w = copy(rb)
    ws = w.get_sheet(index)
         
	#w = Workbook()
	#ws = w.add_sheet('SVM')
    ws.write_merge(item,item,0,len(data[0]),str(label[nameFlower]+1)+': '+nameFlower,style)

    for i in range (len(data[0])):
     	 ws.row(item+1).write(i+1, i+1, style2)
    for i in range (len(data)):
    	 ws.row(item+i+2).write(0, i+1, style2)	

    for i in range (len(data)):
        for j in range (len(data[0])):
            ws.write(item+i+2, j+1, data[i][j])

    num = 0
    for i in range (len(data)):
        num+=data[i][label[nameFlower]]
    ws.write(item+len(data)+2, 0, str(num/numFlower))

    w.save(nameFile)
    w.save(TemporaryFile())


def saveFile2(data, nameFlower, numFlower, pathList, item, index, nameFile, outTrueFile, outFalseFile, timeExecute):
    style = easyxf(
    'pattern: pattern solid, fore_colour red;'
    'align: vertical center, horizontal center;'
    )
    style1 = easyxf(
    'align: vertical center, horizontal center;'
     )

    style2 = easyxf(
    'pattern: pattern solid, fore_colour green;'
    'align: vertical center, horizontal center;'
    )

    style3 = easyxf(
    'pattern: pattern solid, fore_colour blue;'    
    'align: vertical center;'
    )

    rb = open_workbook(nameFile,formatting_info=True)
    w = copy(rb)
    ws = w.get_sheet(index)
    #ws.write_merge(item,item,0,len(data[0]),nameFlower,style3)
    ws.write(item, 0, nameFlower, style3)
    for i in range (len(data[0])):
     	 ws.row(item+2).write(i, i+1, style1)
    
    numTrue = 0
    numWrong = 0

    for i in range (len(data)):
        for j in range (len(data[0])):
            if(data[i][j] == label[nameFlower]):
                numTrue+=1
                ws.write(item+i+3, j, str(data[i][j]), style2)
                ws.write(item+i+4, j, str(timeExecute[j]))
                value = data[i][j]
                #value = value[0]
                #nameFlow = get_name[value]
                #nameFlow = os.path.split(os.path.dirname(pathList[j]))[1]
                #name = os.path.basename(pathList[j])
                #name = name[1:3]+str(numTrue)+'.jpg'
                #orgImg = cv2.imread(pathList[j], cv2.IMREAD_GRAYSCALE)
                #pathSave = os.path.join(outTrueFile, 'count', nameFlow) 
                #if not os.path.exists(pathSave):
                #    os.makedirs(pathSave)
                #pathSave = os.path.join(pathSave, name)    
                #cv2.imwrite(pathSave,orgImg)                    
                                
            else:
                numWrong+=1
                ws.write(item+i+3, j, str(data[i][j]), style)
                ws.write(item+i+4, j, str(timeExecute[j]))
                value = data[i][j]
                #value = value[0]
                #nameFlow = get_name[value]
                #nameFlow = os.path.split(os.path.dirname(pathList[j]))[1]
                #name = os.path.basename(pathList[j])
                #name = name[1:3]+str(numWrong)+'.jpg'
                #orgImg = cv2.imread(pathList[j], cv2.IMREAD_GRAYSCALE)
                #pathSave = os.path.join(outFalseFile, 'count', nameFlow) 
                #if not os.path.exists(pathSave):
                #    os.makedirs(pathSave)
                #pathSave = os.path.join(pathSave,name)
                #cv2.imwrite(pathSave,orgImg)                    

    average = 0
    for timeEx in timeExecute:
        average += timeEx
    average = average/len(data[0])
    ws.write(item+1, 0, str(numTrue/numFlower))
    ws.write(item+1, 2, str(average))
    w.save(nameFile)
    w.save(TemporaryFile())
        
def saveFile3(data, listNameFlower, index, nameFile):
    rb = open_workbook(nameFile,formatting_info=True)
    w = copy(rb)
    ws = w.get_sheet(index)

    for i in range (len(label)):
         ws.row(1).write(i+1,str(get_name[i]))

    for i in range (len(label)):
     	 ws.row(i+2).write(0,str(get_name[i]))

    for i in range (len(data)):
        for j in range (len(data[0])):
            ws.write(label[listNameFlower[i]]+2, j+1, str(data[i][j]))

    w.save(nameFile)
    w.save(TemporaryFile())
