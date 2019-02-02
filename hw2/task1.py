import numpy as np
import cv2
import math
import sys
from threading import Thread
# this method extract image by kernelsize
def extract(row, col, img,kernelSize):
    Timg = np.zeros((kernelSize,kernelSize),np.uint8) # imagin kenarlarına sıfır ekler
    for i in range(kernelSize): #
        for j in range(kernelSize):
            Timg[i][j] = img[row+i][col + j]

    return Timg

def convolution(image,kernel):
    appendSize = len(kernel) -1
    row,col = image.shape
    newImage = np.zeros(((row+appendSize,col+appendSize)),np.uint8)
    newRow,newCol = newImage.shape
    #####  below statements zero append  #############
    for i in range(newRow):
        for j in range(newCol):
            if i < appendSize or j < appendSize:
                newImage[i][j] = 0
            elif i>row or j > col:
                newImage[i][j] = 0
            else:
                newImage[i][j] = image[i-appendSize][j-appendSize]
    #####################################################################################
    resultImage = np.zeros((newRow,newCol),np.uint8)
    for i in range(newRow - appendSize):
        for j in range(newCol - appendSize):
            # extract üzerindeki kernel degerlerinin çıkarır
            Timg = extract(i,j,newImage,appendSize+1)
           # print(Timg)
            #print('#######################')
            def sumF (x):
                if x < 0:
                    return 0
                elif x > 255:
                    return 255
                else:
                    return x
            # kernel ile direk çarpıp sonucu toplar
            sum = sumF(np.sum(np.multiply(Timg,kernel)))
            #print(sum)

            resultImage[i+1][j+1] = sum
    #print(newImage.shape)
    return resultImage

# fourier için kernelı genişletir
def paddingKernel(img,kernel):
    paddedColsize = img.shape[0] - kernel.shape[0]
    paddedRowsize = img.shape[1] - kernel.shape[1]
    addleft = int((paddedColsize) / 2)
    addright = paddedColsize - addleft

    addTop = int((paddedRowsize) / 2)
    addBottom = paddedRowsize - addTop
    # kernela sıfır ekler
    fkernel = np.pad(kernel, ((addleft, addright), (addTop, addBottom)), mode='constant', constant_values=0)
    return fkernel

# fourier kullanan method
def tactic2Conv(image,kernel):
    row,col = image.shape
    print(row,col)
    fkernel = paddingKernel(image, kernel) # padding kernel to image
    if row > 32 or col> 32: # 32 pixelden büyük ise fast fourier uygula
        fimg = np.fft.fft2(image) # fourier of image
        fkernel = np.fft.fft2(fkernel) # fourier of kernel
        nimg = productDirect(fimg, fkernel)
        nimg = np.fft.ifft2(nimg)
        nimg = np.fft.fftshift(nimg)  # aligned orign
        nimg = normMatrix(nimg)
        return nimg
    else: # değil ise normal fourier uygula
        print("my DFT")
        fimg = FDFT(img)
        fkernel = FDFT(fkernel)
        nimg = productDirect(fimg,fkernel)
        nimg = IDFT(nimg)
        nimg = np.fft.fftshift(nimg)  # aligned orign
        nimg = normMatrix(nimg)
        return nimg

# ınverse fourier kernel ile image çarpar
def productDirect(img1,img2):
    row,col = img1.shape
    Timg = np.zeros((row,col),np.complex)
    for i in range(row):
        for j in range(col):
            Timg[i][j] = img1[i][j]*img2[i][j]
    return Timg
# inverse matrix de resim değerini bulmak için kullanılır
def normMatrix(img):
    row,col = img.shape
    Timg = np.zeros((row,col),np.uint8)
    for i in range(row):
        for j in range(col):
            val = img[i][j]
            val = np.absolute(val)
            if val > 255:
                val = 255
            Timg[i][j] = val
    return Timg



# Inverse fourier img
def IDFT(arr):
    row,col = arr.shape
    Timg = np.zeros((row,col),np.complex)
    for u in range(row):
        for v in range(col):
            for i in range(row):
                for j in range(col):
                    Timg[u][v] += arr[i][j] *np.exp(2j * math.pi *((i*u/row) + (j*v/col)))
    Timg = Timg/(row*col)
    return Timg.round(2)

# benim implement ettigim fourier
def FDFT(img):
    row, col = img.shape
    Timg = np.zeros((row, col), np.complex)
    pi = math.pi

    for u in range(row):
        for v in range(col):
            for i in range(int(row)):
                for j in range(int(col/2)):
                    Timg[u][v] += img[i][j*2] * np.exp(-2j * pi * ((i * u / row) + (j*2 * v / col)))

            for i in range(int(row)):
                for j in range(int(col/2)):
                    Timg[u][v] += img[i][j*2+1] * np.exp(-2j * pi * (((i)* u / row) + ((j*2 +1)* v / col)))
        print(u)
    return Timg.round(2)






def main():
    print(sys.argv)
    if len(sys.argv) != 3:
        print("argv python3 ./task1.py image kernel")
        return 0

    path = sys.argv[1]
    try:
        kernel2 = np.loadtxt('kernel.txt',delimiter=',')
    except:
        pass
    try:
        kernel2 = np.loadtxt(sys.argv[2])
    except:
        pass

    kernel2 = np.flip(kernel2,0)
    kernel2 = np.flip(kernel2,1)
    print(kernel2)

    print('######### Tactic of convolution teorem  ####################')
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)


    img = tactic2Conv(img,kernel2)
    cv2.imshow('tactic of convolution theorem', img)
    cv2.waitKey(0)

    print('######### Normal convolution  ####################')

    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    img = convolution(img,kernel2)
    cv2.imshow('Convolution', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()