import cv2
import os
import math
import numpy as np
from setting import BINARY_PATH

def eclipseDistance(point1, point2):
    return math.sqrt(math.pow(point2[0]-point1[0],2)+math.pow(point2[1]-point1[1],2)+math.pow(point2[2]-point1[2],2))

def getImgMax(imgBinary):
    Q = []
    result = []

    size = img.shape
    A = np.zeros(size, dtype=np.uint8)
    B = np.zeros(size, dtype=np.uint8)
    C = np.zeros(size, dtype=np.uint8)
    D = np.zeros(size, dtype=np.uint8)
    Q = findLinkPixel(imgBinary)
    maxIndex = countMaxPixel(Q)
    print(len(Q))
    for item in Q[maxIndex]:
        B[item] = 255
    cv2.bitwise_not(B, C)
    Q2 = findLinkPixel(C)
    maxIndex = countMaxPixel(Q2)
    for item in Q2[maxIndex]:
        C[item] = 0
    C1 = cv2.bitwise_or(B,C)
        

def getKeyPointEdge(imgBinary):
    keyArr = []
    kernel = np.ones((3, 3), np.uint8)
    A = cv2.erode(imgBinary,kernel,iterations = 1)
    imgEgde = cv2.bitwise_xor(imgBinary, A)
    rows, cols = imgEgde.shape
    for x in range(rows):
        for y in range(cols):
            if(imgEgde[x][y]==255):
                keyArr.append((x,y))
    return keyArr


def updateKeyPoint(maskArr, imgGray):
    directionX = [-1, 0, 1,-1, 1,-1, 0, 1]
    directionY = [-1,-1,-1, 0, 0, 1, 1, 1]
    keyArray = getKeyPointEdge(maskArr)
    tmpArr = []
    while(len(keyArray)>0):
        for item in keyArray:
            for direc in range(8):
                X = item[0]+directionX[direc]
                Y = item[1]+directionY[direc]
                if(X<0 or Y<0 or X>=100 or Y>=100 or (maskArr[X][Y]==255)):
                    continue
                #if( abs(imgGray[item]-imgGray[X][Y]) < 10):
                if(eclipseDistance(imgGray[X][Y], imgGray[item])<50):
                    maskArr[X][Y]=255
                    tmpArr.append((X,Y))         
        keyArray.clear()
        keyArray.extend(tmpArr)
        tmpArr.clear()
                        

def findCheckPoints(row, colsLength):
    length = len(row)
    rowCheck = []

    if(length == 0):
        return rowCheck

    head = row[0]
    last = row[length-1]

    if(head[0]>0):
        rowCheck.append((head[0]-1,head[1]-1))
    for item in row:
        rowCheck.append((item[0]-1, item[1]))
    if(last[0] < colsLength-1):
        rowCheck.append((last[0]-1,last[1]+1))
    return rowCheck


def getPoint(imgMat, inRow):
    rows, columns = imgMat.shape
    Q = []
    Result = []
    open = False
    for x in range(columns):
        if(imgMat[inRow][x]>200 and x!=(columns-1)):
            Q.append((inRow,x))
            open = True
        elif((open == True) and (imgMat[inRow][x]==0)):
            Result.append(Q[:])    
            Q.clear()
            open = False
        elif(x==(columns-1) and imgMat[inRow][x]>200):
            Q.append((inRow,x))
            Result.append(Q[:])
            Q.clear()
            open = False
        else:
            open = False
    return Result


def checkRow(rowCheck, rowCluster):
    for checkPoint in rowCheck:
        for point in rowCluster:
            if ((checkPoint[0] == point[0] and checkPoint[1] == point[1])):
                return True
    return False

def getRow(mat, index):
    result = []
    for item in mat:
        if(item[0] == index):
            result.append(item)
    return result

def mergeValue(mat1, mat2, index):
    result = []
    for k in range(index):
        row1 = getRow(mat1, k)
        row2 = getRow(mat2, k)
        result.extend(row1)
        result.extend(row2)
    return result

def changeQueue(matQ, checkR ,checkRowQueue):
    for index, value in enumerate(checkRowQueue):
        if index == 0:
            continue
        value = matQ[index]
        matQ[checkRowQueue[0]].extend(matQ[checkRowQueue[index]])
        #checkRowQueue[0] = mergeValue(checkRowQueue[0], matQ[checkRowQueue[index]])
    matQ[checkRowQueue[0]].extend(checkR)
    del checkRowQueue[0]
    removeElement(matQ, checkRowQueue)
    

def findLinkPixel(img):
    Q = []
    rows, cols = img.shape

    for rowIndex in range(rows):
        points = getPoint(img, rowIndex)
        if(len(points)==0 ):
            continue
        elif(len(Q)==0):
            Q = points
            continue
        for checkR in points:
            checkPoints = findCheckPoints(checkR, cols)
            checkRowQueue = []
            for indexQ, checkQ in enumerate(Q):
                rowQ = getRow(checkQ, rowIndex- 1)
                if(len(rowQ)==0):
                    continue
                elif(checkRow(rowQ, checkPoints)):
                    checkRowQueue.append(indexQ)            
            if (len(checkRowQueue)==0):
                Q.append(checkR)
            elif(len(checkRowQueue)==1):
                Q[checkRowQueue[0]].extend(checkR)
            else:
                changeQueue(Q, checkR, checkRowQueue)
    return Q

def removeElement(matQ ,removeArray):
    matQ = [points for index, points in enumerate(matQ) if index not in removeArray]


def countMaxPixel(matQ):
    max = 0
    maxIndex = 0
    for index, item in enumerate(matQ):
        if(len(item) > max):
            max = len(item)
            maxIndex = index
    return maxIndex


def checkFuction(Q, rows, cols):
    for index in range(rows):
        for index1 in range(len(Q)-1):
            for index2 in range(index1, len(Q)):
                if (checkRow(getRow(Q, index), getRow(Q, index))):
                    return True
    return False
