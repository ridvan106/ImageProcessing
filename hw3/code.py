
""""
Rıdvan Demirci
    141044070

"""
import cv2
import math
import numpy as np
from copy import deepcopy
import threading
BLOCK = 8
# 1.19

# Program to print matrix in Zig-zag pattern

# cift ise sondan tek ise basdan alir
def decodeZigzag(liste):
    matrix = np.zeros((BLOCK,BLOCK)) # matrix to decode
    for i in range(BLOCK):
        for j in range(BLOCK):
            if i+j < len(liste): # if list less than addition of indis
                if liste[i+j] != []: # if list is not empty
                    if (i+j) % 2==0: # çift ise listenin sonundan alir
                        matrix[i][j] = liste[i+j][-1]
                        del liste[i+j][-1] # listenin sonundan siler
                    else:   # tek ise listenin basindan alir
                        matrix[i][j] = liste[i+j][0]
                        del liste[i+j][0]

    return matrix

# whatever happens sum of indis means is on same line
def treverseZigzag(matrix):
    row,col = BLOCK,BLOCK
    zigzag = []
    for i in range(row+col-1): # toplam da row ve coldan 1 eksik olacak
        zigzag.append([])  # liste olusturulur
    for i in range(row):
        for j in range(col):
            if (i+j) %2 == 0: # if sum of indis is even ,it should add first because of order
                zigzag[i+j].insert(0,matrix[i][j])
            else:
                zigzag[i+j].append(matrix[i][j])
    zigzag = list(filter(sum,zigzag))# method of sum all list element
    return zigzag

# calculate DCT
def DCT(img):
    nImg = np.zeros((BLOCK,BLOCK))
    img = np.array(img)
    c1 = 1
    c2 = 1
    for i in range(BLOCK):
        for j in range(BLOCK):
            for k in range(BLOCK):
                for l in range(BLOCK):
                    nImg[i][j] += img[k][l]*math.cos(((2*k+1)*i*math.pi)/(2*BLOCK))*math.cos(((2*l+1)*j*math.pi)/(2*BLOCK))

            if i == 0:
                c1 = 1/math.sqrt(2)
            else:
                c1 = 1
            if j == 0:
                c2 = 1/math.sqrt(2)
            else:
                c2 = 1
            nImg[i][j] = c1*c2*nImg[i][j]/4
    return nImg

# calculate IDCT
def IDCT(img):
    nImg = np.zeros((BLOCK, BLOCK))
    img = np.array(img)
    #print(img)
    c1 = 1
    c2 = 1
    for i in range(BLOCK):
        for j in range(BLOCK):
            for k in range(BLOCK):
                for l in range(BLOCK):
                    if k == 0:
                        c1 = 1 / math.sqrt(2)
                    else:
                        c1 = 1
                    if l == 0:
                        c2 = 1 / math.sqrt(2)
                    else:
                        c2 = 1

                    nImg[i][j] += c1*c2*img[k][l] * math.cos(((2 * i + 1) * k * math.pi) / (2 * BLOCK)) * math.cos(
                        ((2 * j + 1) * l * math.pi) / (2 * BLOCK))


            nImg[i][j] = nImg[i][j] / 4
    return nImg

# calculate mean square error R,B and G
def calculateError(img1,img2):
    row,col,_ = img1.shape
    R1 = img1[:,:,0]
    R2 = img2[:,:,0]
    G1 = img1[:,:,1]
    G2 = img2[:,:,1]
    B1 = img1[:,:,2]
    B2 = img2[:,:,2]
    errorR = 0
    errorG = 0
    errorB = 0
    for i in range(row):
        for j in range(col):
            print(R1[i][j]-R2[i][j])
            errorR += math.pow(R1[i][j]-R2[i][j],2)
            errorG += math.pow(G1[i][j]-G2[i][j],2)
            errorB += math.pow(B1[i][j]-B2[i][j],2)
    LerrorR = errorR/(row*col)
    LerrorG = errorG/(row*col)
    LerrorB = errorB/(row*col)
    print('Layer of Error R :',LerrorR)
    print('Layer of Error G :',LerrorG)
    print('Layer of Error B :',LerrorB)
    print('Total error:',(errorR+errorG+errorB)/(3*row*col))

# product direcly
def productDirec(mat1,mat2):
    mat1 = np.array(mat1)
    mat2 = np.array(mat2)
    for i in range(BLOCK):
        for j in range(BLOCK):
            mat1[i][j] = mat1[i][j]*mat2[i][j]
    return mat1
# dive cell by cell
def divideDirecly(mat1,mat2):
    mat1 = np.array(mat1)
    mat2 = np.array(mat2)
    temp = np.zeros((8,8))
    for i in range(BLOCK):
        for j in range(BLOCK):
            temp[i][j] = mat1[i][j]/mat2[i][j]
    return temp
# Convert RGB to YUV
def convertYUV(img):
    R = np.array(img[:,:,0])
    G = np.array(img[:,:,1])
    B = np.array(img[:,:,2])

    Y = (0.299 * R) + (0.587 * G) + (0.114 * B)

    U =  (R-Y)*0.713  + 128
    V =(B-Y)*0.564 + 128

    Nimg = np.zeros(img.shape,dtype=np.uint8)
    U = np.round(U)
    Y = np.round(Y)
    V = np.round(V)

    Nimg[:, :, 0] = Y
    Nimg[:, :, 1] = U
    Nimg[:, :, 2] = V

    return Nimg
"""def convertRGB(img):
    Y = deepcopy(np.array(img[:, :, 0]))
    U = deepcopy(np.array(img[:, :, 1]))
    V = deepcopy(np.array(img[:, :, 2]))

    R = Y + 1.403*(V - 128)
    G = Y - 0.714*(V - 128) - (0.344*(U - 128))
    B = Y + 1.773*(V - 128)

    R[R<0]=R[R<0]+180
    #R[R>255] =255
    #####################
    G[G<0] = G[G<0] +180
    #G[G>255] = 255
    ######################
    B[B<0] =B[B<0]+180
    #B[B>255] =255
    Nimg = np.zeros(img.shape,dtype=np.int)
    Nimg[:, :, 0] = R
    Nimg[:, :, 1] = G
    Nimg[:, :, 2] = B
    return Nimg"""
# Get block of image 8X8
def getBlock(img,i,j):
    Timg = np.zeros((BLOCK,BLOCK),dtype=np.int)
    row,col = img.shape
    for k in range(0,BLOCK):
        for l in range(0,BLOCK):
            if k+i < row and j+l < col:
                Timg[k][l] = img[k+i][j+l]

    return Timg

# set block to image specific indis
def setBlock(img,block,i,j):
    row, col = img.shape
    for k in range(0,BLOCK):
        for l in range(0,BLOCK):
            if k + i < row and j + l < col:
              img[k+i][l+j] = block[k][l]

    return img

def decodeparelel(huffTable,Q,layer):
    r, c, _ = img.shape
    row = len(huffTable) * BLOCK
    col = len(huffTable[0]) * BLOCK
    #print(huffTable)
    i = 0
    while (i < row):
        j = 0
        while j < col:
            print(i, j)
            block = decodeZigzag(huffTable[round(i / BLOCK)][round(j / BLOCK)])
            ##############################################################

            ################## Product Quatization #######################
            res = np.multiply(block, Q)
            ################## Take IDCT and add128 #######################
            res = IDCT(res) + 128
            ############# Remove Trashold
            res[res < 0] = 0
            res[res > 255] = 255
            ################################
            layer = setBlock(layer, res, i, j)

            j += BLOCK
        i += BLOCK

# this function make decode to encoded image
def decode(huf1,huf2,huf3,Q):
    r,c,_ = img.shape
    Y = np.zeros((r,c))
    U = np.zeros((r,c))
    V = np.zeros((r,c))
    print(len(huf1),len(huf1[0]),len(huf2),len(huf2[0]),len(huf3),len(huf3[0]))
    thread1 = threading.Thread(decodeparelel(huf1,Q,Y)) # decode taken table
    thread2 = threading.Thread(decodeparelel(huf2,Q,U))
    thread3 = threading.Thread(decodeparelel(huf3,Q,V))

    thread1.start()
    thread2.start()
    thread3.start()

    thread1.join()
    thread2.join()
    thread3.join()

    returnedImg = np.zeros((img.shape), dtype=np.uint8)
    returnedImg[:, :, 0] = Y
    returnedImg[:, :, 1] = U
    returnedImg[:, :, 2] = V
    return returnedImg
# this function make encode normal image


def calcuateParelel(layer,table):
    row,col = layer.shape
    i = 0
    while (i < row):
        j = 0
        while j < col:
            block = getBlock(layer, i, j) - 128
            res = divideDirecly(DCT(block), Q)
            temp = treverseZigzag(res)
            table[round(i / BLOCK)][round(j / BLOCK)] = temp
            j += BLOCK
        i += BLOCK

    return table

def compress(img, Q):
    Y = np.asarray(img[:,:,0],dtype=int)
    U = np.asarray(img[:,:,1],dtype=int)
    V = np.asarray(img[:,:,2],dtype=int)
    rowN,colN,_ = img.shape
    zigzagTable1 = np.zeros((math.ceil(rowN/BLOCK),math.ceil((colN/BLOCK))),dtype=object)
    zigzagTable2 = np.zeros((math.ceil(rowN/BLOCK),math.ceil((colN/BLOCK))),dtype=object)
    zigzagTable3 = np.zeros((math.ceil(rowN/BLOCK),math.ceil((colN/BLOCK))),dtype=object)
    thread1 = threading.Thread(target=calcuateParelel,args=(Y,zigzagTable1))
    thread2 = threading.Thread(target=calcuateParelel,args=(U,zigzagTable2))
    thread3 = threading.Thread(target=calcuateParelel,args=(V,zigzagTable3))
    
    thread1.start()
    thread2.start()
    thread3.start()

    thread1.join()
    thread2.join()
    thread3.join()


    return zigzagTable1,zigzagTable2,zigzagTable3


# Quantization matrix
Q = np.matrix('16 11 10 16 24 40 51 61;\
    12 12 14 19 26 58 60 55;\
    14 13 16 24 40 57 69 56;\
    14 17 22 29 51 87 80 62;\
    18 22 37 56 68 109 103 77;\
    24 35 55 64 81 104 103 92;\
    49 64 78 77 103 121 120 101;\
    72 92 95 98 112 100 103 99').astype('float')

img = cv2.imread('horse.jpg') # read image
nimg = deepcopy(img) # copy image to calculateing error
YUVimg = convertYUV(img) # convert RGB to YUV
z1,z2,z3 = compress(YUVimg, Q) # compress return 3 array


print("###########################")

decodeImg = decode(z1,z2,z3,Q) # decode encript decoded image
decodeImg = cv2.cvtColor(decodeImg,cv2.COLOR_YUV2BGR)

cv2.imshow('Decoded Image',decodeImg)

cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.imwrite('horse2.jpg',decodeImg)

calculateError(nimg,decodeImg) # this method calculate error




