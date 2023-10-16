import matplotlib.image as img
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from numpy.lib.arraypad import pad
pic=img.imread("images/4demosaicking.bmp")
plt.imshow(pic)
Arpic=np.array(pic, dtype=np.uint8)
X, Y, Z=Arpic.shape
Xtrans=np.array((X,Y,Z),dtype=np.uint8)
Bayer=np.array((X,Y,Z),dtype=np.uint8)
BGreen=np.zeros((X,Y,Z),dtype=np.uint8)
BRed=np.zeros((X,Y,Z),dtype=np.uint8)
BBlue=np.zeros((X,Y,Z),dtype=np.uint8)
XGreen=np.zeros((X,Y,Z),dtype=np.uint8)
XRed=np.zeros((X,Y,Z),dtype=np.uint8)
XBlue=np.zeros((X,Y,Z),dtype=np.uint8)
print(Arpic.shape)
#Filtr Bayera
BGreen[::2,::2,1]=Arpic[::2,::2,1]
BBlue[1::2,::2,2]=Arpic[1::2,::2,2]
BGreen[1::2,1::2,1]=Arpic[1::2,1::2,1]
BRed[::2,1::2,0]=Arpic[::2,1::2,0]
Bayer=np.uint8(BGreen+BBlue+BRed)
Image.fromarray(Bayer).save("Bayer.jpg")
#Filtr X-trans
XGreen[::6,::6,1]=Arpic[::6,::6,1]
XGreen[::6,3::6,1]=Arpic[::6,3::6,1]
XBlue[::6,1::6,2]=Arpic[::6,1::6,2]
XRed[::6,2::6,0]=Arpic[::6,2::6,0]
XRed[::6,4::6,0]=Arpic[::6,4::6,0]
XRed[1::6,::6,0]=Arpic[1::6,::6,0]
XBlue[2::6,::6,2]=Arpic[2::6,::6,2]
XGreen[3::6,::6,1]=Arpic[3::6,::6,1]
XBlue[4::6,::6,2]=Arpic[4::6,::6,2]
XRed[5::6,::6,0]=Arpic[5::6,::6,0]
XGreen[1::3,1::6,1]=Arpic[1::3,1::6,1]
XGreen[1::3,2::6,1]=Arpic[1::3,2::6,1]
XBlue[1::6,3::6,2]=Arpic[1::6,3::6,2]
XGreen[1::3,4::6,1]=Arpic[1::3,4::6,1]
XGreen[1::3,5::6,1]=Arpic[1::3,5::6,1]
XGreen[2::3,1::6,1]=Arpic[2::3,1::6,1]
XGreen[2::3,2::6,1]=Arpic[2::3,2::6,1]
XRed[2::6,3::6,0]=Arpic[2::6,3::6,0]
XGreen[2::3,4::6,1]=Arpic[2::3,4::6,1]
XGreen[2::3,5::6,1]=Arpic[2::3,5::6,1]
XRed[3::6,1::6,0]=Arpic[3::6,1::6,0]
XBlue[3::6,2::6,2]=Arpic[3::6,2::6,2]
XGreen[3::6,3::6,1]=Arpic[3::6,3::6,1]
XBlue[3::6,4::6,2]=Arpic[3::6,4::6,2]
XRed[3::6,5::6,0]=Arpic[3::6,5::6,0]
XRed[4::6,3::6,0]=Arpic[4::6,3::6,0]
XBlue[5::6,3::6,2]=Arpic[5::6,3::6,2]
XBlue[::6,5::6,2]=Arpic[::6,5::6,2]
Xtrans=XBlue+XGreen+XRed
Image.fromarray(Xtrans).save("Xtrans.jpg")
DeBayer=np.zeros((X,Y,Z),np.uint8)
#Hg i Hr to kernel wynikaj¹cy ze wzoru interpolacji bilinearnej
Hg=np.array([[0,1,0],[1,4,1],[0,1,0]],np.uint8)
Hg=Hg/4
Hr=np.array([[1,2,1],[2,4,2],[1,2,1]],np.uint8)
Hr=Hr/4
for i in range(X):
    print(i)
    for j in range(Y):
        for k in range(Z):
                if i<570:
                    if j<380:
                        #Bilinear interpolation

                        DeBayer[i,j,0]=np.uint8(float(Bayer[i-1,j-1,0]*Hr[0,0]+Bayer[i,j-1,0]*Hr[1,0]+Bayer[i+1,j-1,0]*Hr[2,0]+Bayer[i-1,j,0]*Hr[0,1]+Bayer[i,j,0]*Hr[1,1]+Bayer[i+1,j,0]*Hr[2,1]+Bayer[i-1,j+1,0]*Hr[0,2]+Bayer[i,j+1,0]*Hr[1,2]+Bayer[i+1,j+1,0]*Hr[2,2]))
                        DeBayer[i,j,1]=np.uint8(float(Bayer[i-1,j-1,1]*Hg[0,0]+Bayer[i,j-1,1]*Hg[1,0]+Bayer[i+1,j-1,1]*Hg[2,0]+Bayer[i-1,j,1]*Hg[0,1]+Bayer[i,j,1]*Hg[1,1]+Bayer[i+1,j,1]*Hg[2,1]+Bayer[i-1,j+1,1]*Hg[0,2]+Bayer[i,j+1,1]*Hg[1,2]+Bayer[i+1,j+1,1]*Hg[2,2]))
                        DeBayer[i,j,2]=np.uint8(float(Bayer[i-1,j-1,2]*Hr[0,0]+Bayer[i,j-1,2]*Hr[1,0]+Bayer[i+1,j-1,2]*Hr[2,0]+Bayer[i-1,j,2]*Hr[0,1]+Bayer[i,j,2]*Hr[1,1]+Bayer[i+1,j,2]*Hr[2,1]+Bayer[i-1,j+1,2]*Hr[0,2]+Bayer[i,j+1,2]*Hr[1,2]+Bayer[i+1,j+1,2]*Hg[2,2]))
                        if i==0:
                            DeBayer[i,j,0]=np.uint8(float(Bayer[i,j,0]*Hr[1,1]+Bayer[i+1,j,0]*Hr[2,1]+Bayer[i,j+1,0]*Hr[1,2]+Bayer[i+1,j+1,0]*Hr[2,2]))
                        if j==0:
                            DeBayer[i,j,0]=np.uint8(float(Bayer[i-1,j,0]*Hr[0,1]+Bayer[i,j,0]*Hr[1,1]+Bayer[i+1,j,0]*Hr[2,1]+Bayer[i-1,j+1,0]*Hr[0,2]+Bayer[i,j+1,0]*Hr[1,2]+Bayer[i+1,j+1,0]*Hr[2,2]))

Image.fromarray(DeBayer,'RGB').show()


plt.show()
plt.imshow(Bayer)
plt.show()
plt.imshow(DeBayer)
Image.fromarray(DeBayer).save("demo.jpg")
plt.show()

