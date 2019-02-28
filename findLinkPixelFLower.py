import os
import cv2
import numpy as np

from setting import BINARY_PATH, ORIGIN_PATH, BASE_DIR
from findLink import changeQueue, checkRow, getPoint, findCheckPoints, mergeValue, getRow, removeElement, countMaxPixel, findLinkPixel, updateKeyPoint
    

path = "C:/Users/DUC13T3/Desktop/XLA/Flower/lily/result3"
imgHor = os.path.join(BINARY_PATH, "Hoa Thien Dieu")
path = os.path.join(imgHor, "3str002.png")
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape

Q = []
result = []

size = img.shape
A = np.zeros(size, dtype=np.uint8)
B = np.zeros(size, dtype=np.uint8)
C = np.zeros(size, dtype=np.uint8)
D = np.zeros(size, dtype=np.uint8)

Q = findLinkPixel(img)

maxIndex = countMaxPixel(Q)
print(len(Q))
for item in Q[maxIndex]:
    B[item] = 255

cv2.bitwise_not(B, C)
##print(C)   

Q2 = findLinkPixel(C)

maxIndex = countMaxPixel(Q2)
for item in Q2[maxIndex]:
    C[item] = 0
#A1 = cv2.bitwise_and(src, src, mask = B)
#B1 = cv2.bitwise_and(src, src, mask = C)
C1 = cv2.bitwise_or(B,C)
#D1 = cv2.bitwise_and(src, src, mask = C1)

#maskArr = C1[:]

cv2.imshow("Source", img)


#updateKeyPoint(maskArr, src)

#maskArr = cv2.bitwise_and(src, src, mask = C1)

cv2.imshow("groupMax", C1)


##print(B)
cv2.waitKey(0)             
                                                                                                                                      