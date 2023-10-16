from PIL import Image
import cv2
import math
import numpy as np
import matplotlib.image as img 
def MovingAverage(Arpic):
    X, Y,Z= Arpic.shape
    Mask= np.ones([3,3], dtype=np.uint8)
    Mask= Mask/9
    Newpic= np.zeros([X,Y,Z],dtype=np.uint8)
    
    for i in range(X):
        for j in range(Y):
            if (i!=X-1) and (j!=Y-1):
                   Newpic[i,j,:]=Arpic[i-1,j-1,:]*Mask[0,0]+Arpic[i,j-1,:]*Mask[1,0]+Arpic[i+1,j-1,:]*Mask[2,0]+Arpic[i-1,j,:]*Mask[0,1]+Arpic[i,j,:]*Mask[1,1]+Arpic[i+1,j,:]*Mask[2,1]+Arpic[i-1,j+1,:]*Mask[0,2]+Arpic[i,j+1,:]*Mask[1,2]+Arpic[i+1,j+1,:]*Mask[2,2]
            else:
                   Newpic[i,j,:]=Arpic[i,j,:]
    return Newpic
    
def MedianFilter(Arpic):
    X, Y, Z= Arpic.shape
    Newpic = np.zeros([X,Y,Z],dtype=np.uint8)
    for i in range(X):
        for j in range(Y):
            for k in range(Z):
                if (i<X-2) and (j<Y-2):
                    New=[
                    Arpic[i-2,j-2,k],
                    Arpic[i-2,j+2,k],
                    Arpic[i-2,j+1,k],
                    Arpic[i-2,j-1,k],
                    Arpic[i-2,j,k],
                    Arpic[i-1,j-2,k],
                    Arpic[i-1,j+2,k],
                    Arpic[i-1,j-1,k],
                    Arpic[i-1,j,k],
                    Arpic[i-1,j+1,k],                                        
                    Arpic[i,j+1,k],
                    Arpic[i,j-1,k],
                    Arpic[i,j,k],
                    Arpic[i,j+2,k],
                    Arpic[i,j-2,k],
                    Arpic[i+1,j,k],
                    Arpic[i+1,j-1,k],
                    Arpic[i+1,j+2,k],
                    Arpic[i+1,j-2,k],
                    Arpic[i+1,j+1,k],
                    Arpic[i+2,j-2,k],
                    Arpic[i+2,j-1,k],
                    Arpic[i+2,j,k],
                    Arpic[i+2,j+1,k],
                    Arpic[i+2,j+2,k]
                    ]
                    New=sorted(New)
                    Newpic[i,j,k]=New[12]
                else:
                    Newpic[i,j,k]=Arpic[i,j,k]
    return Newpic

def Gaussian(D,x,y):
    return np.exp(-((x**2+y**2)/(2*D**2)))

def BilateralFilter(Arpic, DS, DR):
    X, Y, Z= Arpic.shape
    Newpic = np.zeros([X,Y,Z],dtype=np.uint8)
    Newpic = np.zeros([X,Y,Z],dtype=np.uint8)
    for px in range(X):
        for py in range(Y):
            for k in range(Z):
                fr=0
                wr=0
                gs=0
                ws=0
                Ip=(Arpic[px,py,1]+Arpic[px,py,2]+Arpic[px,py,0])/3
                for i in range(-1,2):
                    for j in range(-1,2):
                            qx=np.max([0,np.min([X-1,px+i])])
                            qy=np.max([0,np.min([Y-1,py+j])])
                            g=1.0 / (2 * math.pi * (DS ** 2)) * math.exp(- ((qx - px)**2 + (qy - py)**2) / (2 * DS ** 2))
                            gs+=g*Arpic[qx,qy,k]
                            ws+=g
                            Iq=(Arpic[qx,qy,1]+Arpic[qx,qy,2]+Arpic[qx,qy,0])/3
                            ga=1.0 / (2 * math.pi * (DR ** 2)) * math.exp(- ((Iq-Ip)**2) / (2 * DR ** 2))
                            fr+=ga*Arpic[qx,qy,k]*g
                            wr+=ga*g
                Newpic[px,py,k]=int(round((fr)/(wr)))
    return Newpic
def MSE(before,after):
    return np.sqrt((before-after)).mean()
pic=img.imread("images/Leosia.jpg").astype(np.uint8)
clean=img.imread("images/Leoclean.jpg")
Image.fromarray(pic).show()
A=MovingAverage(pic)
Image.fromarray(A).show()
Image.fromarray(A).save("Mean.jpg")
t=cv2.subtract(pic, A)
Image.fromarray(t).save("MeanNoise.jpg")
Image.fromarray(t).show()
B=MedianFilter(pic)
Image.fromarray(B).save("Median.jpg")
Image.fromarray(B).show()
t=cv2.subtract(pic,B)
Image.fromarray(t).save("MeadianNoise.jpg")
Image.fromarray(t).show()
C=BilateralFilter(pic,200,200)
Image.fromarray(C).show()
t=cv2.subtract(pic,C)
Image.fromarray(t).show()
Image.fromarray(C).save("Bilateral.jpg")
Image.fromarray(t).save("BilateralNoise.jpg")
print("AMSE:"+str(MSE(clean,A)))
print("MMSE:"+str(MSE(clean,B)))
print("BMSE"+str(MSE(clean,C)))
