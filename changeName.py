from transformImg import getTopImg, getRGBvalue, checkSame, getCentroid
from setting import BASE_DIR
from findLink import findLinkPixel, countMaxPixel
from helper import list_item
from setting import GRAY_PATH, BINARY_PATH, EDGE_PATH, COLOR_PATH, BASE_DIR
import os
import cv2
import numpy as np

binaryStr = os.path.join(BASE_DIR, "Build", "Binary", "Hoa Thien Dieu")
corlorStr = os.path.join(BASE_DIR, "Build", "Color", "Hoa Thien Dieu")
originStr = os.path.join(BASE_DIR, "Build", "Origin", "Hoa Thien Dieu")
resultStr = os.path.join(BASE_DIR, "Build", "Result", "Hoa Thien Dieu")

path = "C:/Users/DUC13T3/Desktop/Result"
path2 = "C:/Users/DUC13T3/Desktop/Result1"
path3 = "C:/Users/DUC13T3/Desktop/Hoa Thien Dieu"
path4 = "C:/Users/DUC13T3/Desktop/Result2"
path5 = "C:/Users/DUC13T3/Desktop/TriSiDa"


# def nameCode(code, path):
#     for item_path in list_item(path):
#         name = os.path.split(os.path.dirname(item_path))[1]
#         src1 = os.path.basename(item_path)
#         item2 = code+src1[1:]
#         src2 = os.path.join(path,name, item2)
#         os.rename(item_path, src2)

# nameCode('1',COLOR_PATH)
A = [1, 2, 3, 4, 5, 6]

indices = []
indices.append(1)
indices.append(0)
indices.append(3)
somelist = [i for j, i in enumerate(A) if j not in indices]

print(somelist)

# Q = []
# result = []
# path_save = "Cluster"
# for index,item_path in enumerate(list_item(BINARY_PATH)):
#     print(item_path)
#     image = cv2.imread(item_path,cv2.IMREAD_GRAYSCALE)
#     size = image.shape
#     A = np.zeros(size, dtype=np.uint8)
#     B = np.zeros(size, dtype=np.uint8)
#     C = np.zeros(size, dtype=np.uint8)
#     D = np.zeros(size, dtype=np.uint8)
#     name = os.path.split(os.path.dirname(item_path))[1]
#     path = os.path.join(BASE_DIR, path_save, name)
#     if not os.path.exists(path):
#         os.makedirs(path)
#     rows, cols = image.shape
#     Q = findLinkPixel(image)
#     maxIndex = countMaxPixel(Q)

#     for item in Q[maxIndex]:
#         B[item] = 255
#     cv2.bitwise_not(B, C)
#     Q2 = findLinkPixel(C)
#     maxIndex = countMaxPixel(Q2)

#     for item in Q2[maxIndex]:
#         C[item] = 0

#     C1 = cv2.bitwise_or(B,C)
#     file_read = os.path.basename(item_path)
#     img_gray = os.path.join(GRAY_PATH, name, '2'+file_read[1:])
#     src = cv2.imread(img_gray, cv2.IMREAD_GRAYSCALE)
#     maskArr = cv2.bitwise_and(src, src, mask = C1)
#     file_name_save = os.path.join(path, name+str(index)+".png")
#     cv2.imwrite(file_name_save, maskArr)
#     