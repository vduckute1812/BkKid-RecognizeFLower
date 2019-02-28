import os
import cv2
import math
from setting import GRAY_PATH, EDGE_PATH, OUT_PATH, ORIGIN_PATH, COLOR_PATH
from helper import list_item
import numpy as np


def dist(x1,y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)


def moment( img, p, q):
    m=0
    rows, cols = img.shape
    for x in range(rows):
        for y in range(cols):
            if(img[x,y]>50):
                m+=math.pow(x,p)*math.pow(y,q)
    return m

def getCentroid(img):
    m00=moment(img,0,0)
    m01=moment(img,0,1)
    m10=moment(img,1,0)
    y = m10/m00
    x = m01/m00
    return x,y

def check(img):
    print(getCentroid(img))

def GrayToBinary(path_folder):
    for item, path in enumerate(list_item(GRAY_PATH)):
        name = os.path.split(os.path.dirname(path))[1]
        newPath = os.path.join(path_folder,name)
        if not os.path.exists(newPath):
            os.makedirs(newPath)
        dir = os.path.basename(path)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        newPath = os.path.join(newPath,dir)
        ret,new_image = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.imwrite(newPath,new_image)

def SunToGray(path_folder):
    for item, path in enumerate(list_item(COLOR_PATH)):
        name = os.path.split(os.path.dirname(path))[1]
        newPath = os.path.join(path_folder,name)
        if not os.path.exists(newPath):
            os.makedirs(newPath)
        dir = os.path.basename(path)        
        newPath = os.path.join(newPath,dir)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(newPath,img)        

def GrayToEgde(path_folder):
    for item, path in enumerate(list_item(GRAY_PATH)):
        name = os.path.split(os.path.dirname(path))[1]
        newPath = os.path.join(path_folder,name)
        if not os.path.exists(newPath):
            os.makedirs(newPath)
        dir = os.path.basename(path)
        newPath = os.path.join(newPath,dir)
        print(newPath)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        new_image = cv2.Canny(img,100,200)
        cv2.imwrite(newPath,new_image)

def getTop(img):                        #y
    rows, cols = img.shape
    for x in range(rows):
        for y in range(cols):
            if(img[x,y]>200):
                return x
    return 0

def getTopImg(img):                        #x, y
    rows, cols = img.shape
    for x in range(rows):
        for y in range(cols):
            if(img[x,y]>200):
                return x, y
    return 0

def getRGBvalue(img, x, y):    
    R = img[x][y][0]
    G = img[x][y][1]
    B = img[x][y][2]
    return R, G, B

def checkSame(img1, img2, x, y):
    R1, G1, B1 = getRGBvalue(img1, x, y)
    R2, G2, B2 = getRGBvalue(img2, x, y)
    if(R1==R2 and G1==G2 and B1==B2):
        return True
    else:
        return False

def getBot(img):                        #h
    rows, cols = img.shape
    for x in range(rows-1,-1,-1):
        for y in range(cols):
            if(img[x,y]>200):
                return x
    return rows-1
   
def getLeft(img):                       #x
    rows, cols = img.shape
    for y in range(cols):
        for x in range(rows):
            if(img[x,y]>200):
                return y
    return 0

def getRight(img):                       #w
    rows, cols = img.shape
    for y in range (cols-1,-1,-1):
        for x in range(rows):
            if(img[x,y]>200):
                return y
    return cols-1

def checkImg(img):
    centroidX,centroidY = getCentroid(img)      #x truc ngang, y truc doc
    print(centroidX, centroidY)

def oriToScale(path):
    img = cv2.imread(path)
    row, col, channel = img.shape
    sc = col/row
    if(sc>1):
        x = col-row
        crop_image= cv2.copyMakeBorder(img,int(x/2),int(x/2),0,0,cv2.BORDER_CONSTANT,value= 0)
    if(sc<1):
        x = row-col
        crop_image= cv2.copyMakeBorder(img,0,0,int(x/2),int(x/2),cv2.BORDER_CONSTANT,value= 0)
    crop_image = cv2.resize(crop_image,(100,100))
    path2 = "C:/Users/DUC13T3/Documents/Visual Studio 2013/Projects/FlowerDetect-Core/hoarum2.jpg"
    cv2.imwrite(path2,crop_image)

def BoundingRect(binaryImg, edgeImg, pathSave):
    #Img,contours,hierarchy = cv2.findContours(fileImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ##nameFile = os.path.basename(fileImg)
    ## Find the index of the largest contour
    #areas = [cv2.contourArea(c) for c in contours]
    #max_index = np.argmax(areas)
    #cnt=contours[max_index]
    #x,y,w,h = cv2.boundingRect(cnt)
    x = getLeft(binaryImg)   #left
    y = getTop(binaryImg)     #top
    w = getRight(binaryImg)  #right
    h = getBot(binaryImg)     #bot
    centroidX,centroidY = getCentroid(edgeImg)      #x truc ngang, y truc doc
    name = os.path.split(pathSave)[1]
    #print(name)
    #print((x,y,w,h))
    #print((centroidX, centroidY))
    dleft  = centroidX - x         #Left
    dbot   = h - centroidY         #Bot
    dtop   = centroidY - y         #Top
    dright = w - centroidX         #Right
    max = 0
    for i in (dleft,dbot,dtop,dright):
        if(i>max):
            max=i

    #print(max)                         
    crop_image = edgeImg[y:h, x:w]
    row, col = crop_image.shape
    #print(x,y,w,h)
    #print((max-dtop,max-dbot,max-dleft,max-dright))

    crop_image= cv2.copyMakeBorder(crop_image,int(max-dtop),int(max-dbot),
                                   int(max-dleft),int(max-dright),cv2.BORDER_CONSTANT,value= 0)

    #sc = col/row
    #if(sc>1):
    #    x = col-row
    #    crop_image= cv2.copyMakeBorder(crop_image,int(x/2),int(x/2),0,0,cv2.BORDER_CONSTANT,value= 0)
    #if(sc<1):
    #    x = row-col
    #    crop_image= cv2.copyMakeBorder(crop_image,0,0,int(x/2),int(x/2),cv2.BORDER_CONSTANT,value= 0)
    crop_image = cv2.resize(crop_image,(50,50))
    #for i in crop_image:
    #    for j in crop_image:
    rows, cols = crop_image.shape
    for i in range(rows):
        for j in range(cols):
            if(crop_image[i,j]>100):
               crop_image[i,j] = 255
            else:
                crop_image[i,j] = 0
    checkImg(crop_image)
    cv2.imwrite(pathSave,crop_image)


def TransformImg(path_list, pathSave):
    for item, path in enumerate(list_item(path_list)):
        name = os.path.split(os.path.dirname(path))[1]
        newPath = os.path.join(pathSave,name)
        if not os.path.exists(newPath):
            os.makedirs(newPath)
        dir = os.path.basename(path)
        edgePath = os.path.join(EDGE_PATH, name,dir)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(edgePath, cv2.IMREAD_GRAYSCALE)
        newPath = os.path.join(newPath,dir)
        BoundingRect(img, img2, newPath)
        
def keepPistil(path1):
    binaryImg = cv2.imread(path1,cv2.IMREAD_GRAYSCALE)
    centroidX,centroidY = getCentroid(binaryImg)      #x truc ngang, y truc doc

    #if not os.path.exists(savePath):
    #    os.makedirs(savePath)

    XBot= int(centroidY)
    YBot = int(centroidX)
    while(binaryImg[XBot,YBot]<200):
        XBot=XBot+1

    XTop = int(centroidY)  
    YTop = int(centroidX)  
    while(binaryImg[XTop,YTop]<200):
        XTop=XTop-1

    XLeft = int(centroidY)
    YLeft = int(centroidX)
    while(binaryImg[XLeft,YLeft]<200):
        YLeft=YLeft-1

    XRight = int(centroidY)
    YRight = int(centroidX)
    while(binaryImg[XRight,YRight]<200):
        YRight=YRight+1
    
    dT = dist(int(centroidY), XTop, int(centroidX), YTop)
    dB = dist(int(centroidY), XBot, int(centroidX), YBot)
    dL = dist(int(centroidY), XLeft, int(centroidX), YLeft)
    dR = dist(int(centroidY), XRight, int(centroidX), YRight)

    min = 0 
    for i in (dT,dB,dL,dR):
        if(i<min):
            min=i
    print(min)
    return min
    #sizeX = int(centroidY)-int(max)
    #sizeY =  int(centroidX)-int(max)
    #crop_image = originImg[sizeX:sizeX+int(min)*2,sizeY:sizeY+int(min)*2] 
    #nameFile = os.path.basename(path1)
    #saveFile = os.path.join(savePath,nameFile)
    #cv2.imwrite(saveFile,crop_image)
    #s_img = cv2.imread("smaller_image.png", -1)
    #for c in range(0,3):
    #    l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1], c] =  
    #    s_img[:,:,c] * (s_img[:,:,3]/255.0) +  l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1], c] * (1.0 - s_img[:,:,3]/255.0)              

def keepPistil2(path1):    
    binaryImg = cv2.imread(path1,cv2.IMREAD_GRAYSCALE)
        
    centroidX,centroidY = getCentroid(binaryImg)      #x truc ngang, y truc doc -> x cot, y hang
    x = getLeft(binaryImg)    #left
    y = getTop(binaryImg)     #top
    w = getRight(binaryImg)   #right
    h = getBot(binaryImg)     #bot

    dictX = (w-x)/4
    dictY = (h-y)/4
    x=x+int(dictX)
    w=w-int(dictX)
    y=y+int(dictY)
    h=h-int(dictY)
    return x,y,w,h, binaryImg


def mergePistil(binaryPath, savePath):  
    x, y, w, h, img = keepPistil2(binaryPath)

    #img = cv2.imread(binaryPath, cv2.IMREAD_GRAYSCALE)
    for i in range(y, h):
        for j in range(x, w):
            if(img[i,j]<200):
                img[i,j]=255

    #nameFlower = os.path.split(os.path.dirname(binaryPath))[1]
    #print(nameFlower)
    #saveFile = os.path.join(savePath,nameFlower)
    if not os.path.exists(binaryPath):
        os.makedirs(binaryPath)
    basedir = os.path.basename(binaryPath)
    nameFile = "1"+ basedir[1:]
    nameColorImg = os.path.join("C:/Users/DUC13T3/Desktop/XLA/Flower/daisy/result",nameFile)
    print(nameFile)
    
    colorImg = cv2.imread(nameColorImg)
    ret, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #gray_image = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY)
    #saveFile2 =os.path.join(savePath, colorFile)
    res = cv2.bitwise_and(colorImg, colorImg, mask = img)
    saveFile = os.path.join(savePath,nameFile)

    print(saveFile)
    cv2.imwrite(saveFile, res)
    #cv2.imwrite(saveFile2, img)
# 