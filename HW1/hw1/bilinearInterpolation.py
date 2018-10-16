import cv2
import numpy as np
import math
# yeni olusturulan matrix için kordinatlari yerleştirir
def fillFirst(source,dest,sx,sy):
    col,row = source.shape
    for i in range(col):
        for j in range(row):
            dest[i*sy][j*sx] = source[i][j]
    return dest
def fillBorder(source,dest,sx,sy):
    col,row = dest.shape
    print(dest[0][0])
    j=0
    while (j<col):
        i = 0
        while (i<(row-1)):
            if i%sx != 0:
                prev = int(sx * (i/sx))
                next = prev + sx

                decrease = (dest[j][next]-dest[j][prev])/sx
                dest[j][i] = dest[j][i-1]+decrease

            i +=1
        j += sy
###################################################################################
    j = 0
    while (j < row):
        i = 0
        while (i < col-1):
            if i % sy != 0:
                prev = int(sy * i / sy)
                next = prev + sy
                decrease = (dest[next][j] -dest[prev][j])/sy
                dest[i][j] = dest[i-1][j]+ decrease

            i += 1
        j += sx

    return dest
def blinear(dest,sx,sy):
    col, row = dest.shape
    print(dest[0][0])
    j = 0
    while (j < col-1):
        i = 0
        if j % sy != 0:
            while (i < row - 1):
                if i %sx != 0:
                    prev = int(sy * j / sy)
                    next = prev + sy
                    decrease = (dest[next][i] - dest[prev][i]) / sy
                    dest[j][i] = dest[j-1][i]+ decrease
                i += 1
        j += 1

    return dest
def fillblank(img):
    col , row = img.shape
    for i in range(col):
        for j in range(row):
            if j != row-1:
                if img[i][j] == 0 and img[i][j-1] != 0 and img[i][j+1]:
                    img[i][j] = img[i][j-1]
    return img


def rotation(Nimg, theta):
    degree = math.degrees(theta) # radian convert degree
    col,row = Nimg.shape
    print(col,row)
    Timg = np.zeros((col,row),np.uint8)
    x_orgin = col/2
    y_orgin = row/2
    for i in range(col):
        for j in range(row):
            xN = math.cos(theta)*(j-y_orgin) - (i-x_orgin) *math.sin(theta) + y_orgin

            yN =math.sin(theta)*(j- y_orgin) +(i-x_orgin)*math.cos(theta) +x_orgin
            if xN>=0 and round(xN) <row and yN>=0 and round(yN) <col:
                Timg[round(yN)][round(xN)] = Nimg[i][j]
    Timg = fillblank(Timg)
    return Timg

def scale(input,sx,sy,theta):
    img = cv2.imread(input,0)
    col,row = img.shape

    sx = int(round(sx))
    sy = int(round(sy))
    Ncol = col * sx
    Nrow = row * sy
    Nimg = np.zeros([Ncol,Nrow],np.uint8)
    Nimg = fillFirst(img,Nimg,sx,sy)
    Nimg  = fillBorder(img,Nimg,sx,sy)
    Nimg = blinear(Nimg,sx,sy)
    Nimg = rotation(Nimg,theta)
    cv2.imshow('ada',Nimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

scale('valve.png',2,2,1.5707963268)
"""img = cv2.imread('valve.png',0)
Nimg = rotation(img,1.5707963268)
cv2.imshow('ada', Nimg)
cv2.waitKey(0)
cv2.destroyAllWindows()"""
