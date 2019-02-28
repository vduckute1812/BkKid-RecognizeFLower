import numpy as np
import os
from setting import NAME_LIST, OUT_PATH
import math
import cv2

def resize(image):
    height = image.shape[0]
    width = image.shape[1]
    sc = width*1.0/height
    if sc>1:
        x = width-height
        image = cv2.copyMakeBorder(image,int(x/2),int(x/2),0,0,cv2.BORDER_CONSTANT,value=[0,0,0])
    if sc<1:
        x = height-width
        image = cv2.copyMakeBorder(image,0,0,int(x/2),int(x/2),cv2.BORDER_CONSTANT,value=[0,0,0])

    # r = 100.0 / image.shape[0]
    dim = (100, 100)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return image


def kmeans(img,k=3, loop=5):
    # img = cv2.imread('1ant009.jpg')
    # im = cv2.cvtColor(img,cv2.COLOR_RGB2Lab)
    Z = img.reshape((-1,3))

    # convert to np.float32
    # Z = np.float32(Z)
    # height, width , depth = img.shape
    # print img.shape
    # # Merge a* with b*
    # lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # l_channel, a_channel, b_channel = cv2.split(lab)

    # Z1 = a_channel.reshape((height*width,1))
    # Z2 = b_channel.reshape((height*width,1))

    # Z = np.hstack((a_channel, b_channel))

    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, loop, 1.0)
    # criteria = (cv2.TERM_CRITERIA_MAX_ITER, loop, 1.0)
    ret,label,center = cv2.kmeans(Z,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    # cv2.imwrite("ok3.jpg", res2)
    return label, center, res2


def onClusterChange(cluster):
    global num_of_cluster
    num_of_cluster = cluster
    global update
    update = True
    return None

def onKmeanChange(loop):
    global loop_of_kmean
    loop_of_kmean = loop
    global update
    update = True

def onErosionChange(loop):
    global Erosion
    Erosion = loop
    global update
    update = True

def onDilasionChange(loop):
    global Dilasion
    Dilasion = loop
    global update
    update = True


def fill(board,image,m,n):
    for i in xrange(image.shape[0]):
        for j in xrange(image.shape[1]):
            board[m*100+i][n*image.shape[1]+j] = image[i][j]
    return board

def joinMutipleImage(*images):
    m = int(round(len(images)/2.0))
    n = 2
    width = 0
    height = 200
    for i in xrange(m):
        width = width + images[i].shape[1]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    # for item in images:
    k = 0
    for i in xrange(n):
        for j in xrange(m):
            if k< len(images):
                canvas = fill(canvas,images[k],i,j)
                k = k + 1
    return canvas

def extraction(image,n, count):
    result = []
    label, center, kmean_image = kmeans(image,n, count)
    # print "KMEAN IMAGE",type(image_kmean),"IMAGE",type(image)
    # new_image = image.tolist()
    # print "NEW IMAGE",type(new_image)

    # new_image = map(lambda row: map(lambda point: point if point != center[0].tolist() else [0,0,0], row), new_image)
    # print "RESULT IMAGE",type(result)
    # new_image = np.array(new_image, dtype=np.uint8)
    # result.append(new_image)
    for i in xrange(n):
        new_image = image.tolist()
        kmean_image_list = kmean_image.tolist()
        # new_image = map(lambda row: map(lambda point: point if point != center[i].tolist() else [0,0,0], row), kmean_image_list)
        for index_row,row in enumerate(kmean_image_list):
            for index_column,point in enumerate(row):
                if point != center[i].tolist():
                    new_image[index_row][index_column]=[0,0,0]

        new_image = np.array(new_image, dtype=np.uint8)
        # new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
        # ret,new_image = cv2.threshold(new_image,0,255,cv2.THRESH_OTSU)
        result.append(new_image)
    return result

def showImage(image):
    global num_of_cluster
    num_of_cluster = 3
    global loop_of_kmean
    loop_of_kmean = 5
    global Erosion
    Erosion = 0
    global Dilasion
    Dilasion = 0
    image = resize(image)

    # print image
    wname = "Image"
    cv2.namedWindow(wname)

    # create trackbars for color change
    cv2.createTrackbar('Cluster',wname,3,20,onClusterChange)
    cv2.createTrackbar('KCount',wname,5,20,onKmeanChange)
    # cv2.createTrackbar('Erosion',wname,0,1,onErosionChange)
    # cv2.createTrackbar('Dilasion',wname,0,1,onDilasionChange)
    global update
    update = True
    run = True
    while(run):
        # if update:
        images = extraction(image,num_of_cluster, loop_of_kmean)
        show_image = joinMutipleImage(*images)
        cv2.imshow("Image", show_image)
        update = False

        key = cv2.waitKey(0)
        if key > ord('0') and key <= ord('9'):
            return images[key-48-1]
        elif key == ord('q'):
            run = False

def huMonents(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.HuMoments(cv2.moments(image)).flatten()

def logarit(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hu = cv2.HuMoments(cv2.moments(image)).flatten()
    log = list(map(lambda x: math.log(abs(x)), hu))
    return log

def list_item(input_path):
    for item in os.listdir(input_path):
        if item in NAME_LIST:
            path = os.path.join(input_path, item)
            for hoa_item in os.listdir(path):
                if hoa_item.endswith('.png') or hoa_item.endswith('.jpg'):
                    path_hoa = os.path.join(path, hoa_item)
                    yield path_hoa


def save_img(label, code, input, img):
    flower_name = os.path.basename(os.path.dirname(input))
    item_name = os.path.basename(input)
    new_name = code + item_name[1:]
    out_dir = os.path.join(OUT_PATH, label, flower_name)
    out_path = os.path.join(OUT_PATH, label, flower_name, new_name)
    print(out_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cv2.imwrite(out_path, img)

def rename(dir_input,code,index):
    # item_name = os.path.basename(dir_input)
    dir_name = os.path.dirname(dir_input)
    new_name = "0{}{:03}.png".format(code, index)
    dir_name = os.path.join(dir_name,new_name)
    print(dir_name)
    os.rename(dir_input, dir_name)
    # print item_name

def renameEdge(dir_input):
    dir_name = os.path.dirname(dir_input)
    file = os.path.basename(dir_input)
    new_name = "3"+file[1:]
    dir_name = os.path.join(dir_name, new_name)
    os.rename(dir_input, dir_name) 

def renameImgEdge(path):
    for index, item_path in enumerate(list_item(path)):
        renameEdge(item_path)

def nameGrToEgde(dir_input,nameFlower, fileFlower):
    name = "4"+fileFlower[1:]
    path = os.path.join(dir_input,nameFlower, name)
    return path