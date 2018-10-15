import cv2
import numpy as np
import matplotlib.pyplot as plt
#global histogram
def histogramEquation(img):

    Timage = img
    histogram = []
    cumhistogram =[]
    ##################  calculate histogram  #########################3
    for i in range(256):
        histogram.append(0)
        cumhistogram.append(0)
    row,col = img.shape
    #print(row,col)
    for i in range(row):
        for j in img[i]:
            histogram[j] = histogram[j] +1
    ######################### cumulative histogram ############################
    cumPixel = 0
    i=0
    for hist in histogram:
        cumhistogram[i] = cumhistogram[i-1]+hist
        i +=1
    sumPixel = cumhistogram[255]
    #print(histogram)
    cumhistogram = np.array(cumhistogram) / (sumPixel) # cumulative probibilty
    cumhistogram = (cumhistogram * 255).round()
    cumhistogram = cumhistogram.astype(int)

    #print(cumhistogram)
   # print(img)
   # print('####################################')
    for i in range(row):
        for j in range(col):
            img[i][j] = cumhistogram[img[i][j]]
    return img
# #######################  local adaptive histogram #######################
def adaptiveHist(img):
    img = np.array(img)
    col,row= img.shape
    kernel = 8
    strideCol = int(col / kernel)
    strideRow = int(row / kernel)
    print(strideCol)
    region = np.zeros([kernel,kernel],np.uint8)
    counterCol = 0
    counterRow = 0
    col = col -(kernel -1)
    row = row - (kernel -1)
    while counterRow < strideCol:
        while counterCol <strideRow:
            for i in range(kernel):
                for j in range(kernel):
                    region[i][j] = img[i+(counterRow*kernel)][j+(counterCol*kernel)]
            region = histogramEquation(region)
            for i in range(kernel):
                for j in range(kernel):
                    img[i+(counterRow*kernel)][j+(counterCol*kernel)] = region[i][j]
            counterCol +=1
        counterCol = 0
        counterRow +=1
    return img

def task2Adaptive(input):
    img = np.array(cv2.imread(input, cv2.IMREAD_GRAYSCALE))
    print(img,'\n','##################################')
    newImg = adaptiveHist(img)
    cv2.imshow('EquationImage',newImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
hs = task2Adaptive('valve.png')




img = cv2.imread('valve.png',0)
# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)

cv2.imshow('sd',cl1)
cv2.waitKey(0)
cv2.destroyAllWindows()