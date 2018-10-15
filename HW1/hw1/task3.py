import numpy as np
import cv2





# kesisen matrix de çıkarma yapar
def sunbstract(mat1,kernel):
    kernel = np.array(kernel)
    mat1 = np.array(mat1)
    row,col = kernel.shape
    for i in range(row):
        for j in range(col):
            if kernel[i][j] == 0:
                mat1[i][j] = 0
    if mat1.sum() == 0:
        return 0
    res = mat1 - kernel
    if res.sum() == 0:
        return 0
    return 1
def erosion(img, structure):
    kernel, _ = structure.shape
    region = np.zeros([kernel, kernel], int)
    img = np.array(img)
    Timg = np.array(img)
    col, row = img.shape
    colCount = 0
    rowCount = 0
    col = col - (kernel - 1)
    row = row - (kernel - 1)
    while rowCount < col:
        while colCount <row:
            for i in range(kernel):
                for j in range(kernel):
                    region[i][j] = img[i+rowCount][j+colCount]
            result = sunbstract(region,structure)
            if result != 0:
                Timg[rowCount+1][colCount+1] = 0
            colCount += 1
        colCount = 0
        rowCount += 1
    return Timg
def dilation(img, structure):
    kernel,_ = structure.shape
    region = np.zeros([kernel,kernel], int)
    img = np.array(img)
    Timg = np.array(img)
    col, row = img.shape
    colCount = 0
    rowCount = 0
    col = col - (kernel - 1)
    row = row - (kernel - 1)
    while rowCount < col:
        while colCount < row:
            for i in range(kernel):
                for j in range(kernel):
                    region[i][j] = img[i + rowCount][j + colCount]
            result = sunbstract(region, structure)
            if result != 0:
                Timg[rowCount + 1][colCount + 1] = 255
            colCount += 1
        colCount = 0
        rowCount += 1
    return Timg



def opening(input):
    img = np.array(cv2.imread(input, cv2.IMREAD_GRAYSCALE))
    print(img)
    kernel = np.matrix([[255,255,255,255,255],
                        [255,255,255,255,255],
                        [255,255,255,255,255],
                        [255, 255, 255, 255, 255],
                        [255, 255, 255, 255, 255]])
    Timg = erosion(img, kernel)
    Timg = dilation(Timg,kernel)
    cv2.imshow('Remove Artifact',Timg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#opening('abdomen.png')


def numberOfComponent(input):
    img = cv2.imread(input,cv2.IMREAD_GRAYSCALE)
    col, row = img.shape
    count = 0
    Timg = np.zeros((col,row),np.uint8)
    colCounter = 0
    rowCounter = 0
    pixelCount = 1
    while rowCounter < col:
        while colCounter< row:
            if rowCounter != 0 and colCounter != 0:
                if img[rowCounter][colCounter] == 255:
                    if img[rowCounter][colCounter-1] == 0 and img[rowCounter - 1][colCounter] == 0:
                        Timg[rowCounter][colCounter] = pixelCount
                        count += 1
                        pixelCount+=1
                    elif img[rowCounter][colCounter-1] == 0 and img[rowCounter - 1][colCounter] == 255:
                        Timg[rowCounter][colCounter] = Timg[rowCounter -1][colCounter]
                    elif img[rowCounter][colCounter-1] == 255 and img[rowCounter - 1][colCounter] == 0:
                        Timg[rowCounter][colCounter] = Timg[rowCounter][colCounter - 1]
                    elif img[rowCounter][colCounter-1] != 0 and img[rowCounter - 1][colCounter] != 0:
                        if  Timg[rowCounter -1][colCounter] == Timg[rowCounter][colCounter - 1]:
                            Timg[rowCounter][colCounter] = Timg[rowCounter][colCounter - 1]
                        else:
                            Timg[rowCounter][colCounter] = Timg[rowCounter][colCounter - 1]
                            pixelTarget = Timg[rowCounter -1][colCounter]
                            pixelSource = Timg[rowCounter][colCounter-1]
                            Timg[Timg == pixelTarget] = pixelSource # replace element
                            count -=1


            colCounter +=1
        colCounter = 0
        rowCounter +=1
    print(count)
    cv2.imshow('conponent',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
numberOfComponent('openningDeneme.png')





import cv2
import numpy as np

img = cv2.imread('openningDeneme.png',0)
kernel = np.ones((5,5),np.uint8)
#Terosion = cv2.dilate(img,kernel,iterations = 1)
#Terosion = cv2.erode(img,kernel,iterations = 1)
#opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
#cv2.imwrite('eponnigOpencv_kernel5.png', opening)
num,img = cv2.connectedComponents(img,connectivity=4)
cv2.imshow('conponsssent',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(num-1)