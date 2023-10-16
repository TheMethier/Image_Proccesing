import matplotlib.image as img
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import math
from PIL import Image
import time
def NNscale(Arpic, Scale):

    X,Y=Arpic.shape
    Xn=math.floor(X*Scale)
    Yn=math.floor(Y*Scale)
    Newpic=np.empty([Xn,Yn])
    Xratio=float((X)/(Xn))
    Yratio=float((Y)/(Yn))
    for i in range(Xn):
        for j in range(Yn):
            X1,Y2=math.floor(Xratio*i),math.floor(Yratio*j)
            Newpic[i,j]=Arpic[X1,Y2]
    return(Newpic)

def Bilinearscale(Arpic, Scale):
    X,Y=Arpic.shape
    Xn=math.floor(X*Scale)
    Yn=math.floor(Y*Scale)
    Newpic=np.empty([Xn,Yn])
    Xratio=float((X-1)/(Xn-1))
    Yratio=float((Y-1)/(Yn-1))
    for i in range(Xn-1):
        for j in range(Yn-1):
            x1,y1=math.floor(i*Xratio),math.floor(j*Yratio)
            x2,y2=math.ceil(i*Xratio),math.ceil(j*Yratio)
            xw=i*Xratio-x2
            yw=j*Yratio-y2
            q=Arpic[x1, y1]
            w=Arpic[x2, y1]
            e=Arpic[x1, y2]
            r=Arpic[x2, y2]
            New=q*xw*yw+w*(1-xw)*yw+e*xw*(1-yw)+r*(1-xw)*(1-yw)
            Newpic[i][j]=New
    return Newpic

def BicubicScale(Arpic,Scale):
    a=-0.75
    X,Y=Arpic.shape
    Xn=math.floor(X*Scale)
    Yn=math.floor(Y*Scale)
    Newpic=np.zeros([Xn,Yn])
    Xratio=float((X)/(Xn))
    Yratio=float((Y)/(Yn))
    for i in range(Xn-4):
        for j in range(Yn-4):
            x,y=i*Xratio,j*Yratio
            x1=x-math.floor(x)+1
            x2=x-math.floor(x)
            x3=1+math.floor(x)-x
            x4=2+math.floor(x)-x
            y1=1+y-math.floor(y)
            y2=y-math.floor(y)
            y3=1+math.floor(y)-y
            y4=2+math.floor(y)-y
            mat_x=np.matrix([[K(x1,a),K(x2,a),K(x3,a),K(x4,a)]])
            mat_y=np.matrix(
            [[K(y1,a)],[K(y2,a)],[K(y3,a)],[K(y4,a)]])
            mat_m = np.matrix([[Arpic[int(y-y1), int(x-x1)],
                                    Arpic[int(y-y2), int(x-x1)],
                                    Arpic[int(y+y3), int(x-x1)],
                                    Arpic[int(y+y4), int(x-x1),]],
                                   [Arpic[int(y-y1), int(x-x2),],
                                    Arpic[int(y-y2), int(x-x2),],
                                    Arpic[int(y+y3), int(x-x2),],
                                    Arpic[int(y+y4), int(x-x2),]],
                                   [Arpic[int(y-y1), int(x+x3),],
                                    Arpic[int(y-y2), int(x+x3)],
                                    Arpic[int(y+y3), int(x+x3)],
                                    Arpic[int(y+y4), int(x+x3)]],
                                   [Arpic[int(y-y1), int(x+x4)],
                                    Arpic[int(y-y2), int(x+x4)],
                                    Arpic[int(y+y3), int(x+x4)],
                                    Arpic[int(y+y4), int(x+x4),]]])
            Newpic[j,i]=np.dot(np.dot(mat_x,mat_m),mat_y)    
    return Newpic

def NNRotation(Arpic, angle):
        X,Y=Arpic.shape
        ang=float(angle*(math.pi/180))
        Xc=int(X/2)
        Yc=int(Y/2)
        Newpic=np.zeros((X,Y),np.uint8)
        for i in range(X):
            for j in range(Y):
                A=round((i-Xc)*np.cos(ang)+(j-Yc)*np.sin(ang)+Xc)
                B=round(-(i-Xc)*np.sin(ang)+(j-Yc)*np.cos(ang)+Yc)
                if A>0 and B>0 and A<X and B<Y:
                    Newpic[i, j]=Arpic[int(A), int(B)]   
        return Newpic

def BilinearRotation(Arpic, angle): 
    X,Y=Arpic.shape
    ang= np.radians(angle)
    Xc=X//2;
    Yc=Y//2;
    Newpic=np.empty([X,Y])
    for i in range(X):
        for j in range(Y):
            A=((i-Xc)*np.cos(ang)+(j-Yc)*np.sin(ang)+Xc)
            B=(-(i-Xc)*np.sin(ang)+(j-Yc)*np.cos(ang)+Yc)
            T=math.floor(A)
            Q=math.floor(B)
            W=T-A
            R=Q-B
            if(T<X and Q<Y and T>0 and Q>0):
                Newpic[i,j]=(1-W)*(1-R)*Arpic[T,Q]+Newpic[i,j]
            if(T+1<X and Q+1<Y and T+1>0 and Q+1>0):
                Newpic[i,j]=(W)*(R)*Arpic[T+1,Q+1]+Newpic[i,j]
            if(T<X and Q+1<Y and T>0 and Q+1>0):
                Newpic[i,j]=(W)*(1-R)*Arpic[T,Q+1]+Newpic[i,j]
            if(T+1<X and Q<Y and T+1>0 and Q>0):
                Newpic[i,j]=(1-W)*(R)*Arpic[T+1,Q]+Newpic[i,j]
    return Newpic

def BicubicRotation(Arpic,angle):
    a=-0.75
    X,Y=Arpic.shape
    ang= np.radians(angle)
    X,Y=Arpic.shape
    Xc=X//2;
    Yc=Y//2;
    Newpic=np.zeros([X,Y],np.uint8)
    for i in range(X-4):
        for j in range(Y-4):
            x,y=(i-Xc)*np.cos(ang)+(j-Yc)*np.sin(ang)+Xc,-(i-Xc)*np.sin(ang)+(j-Yc)*np.cos(ang)+Yc
            x1=x-math.floor(x)+1
            x2=x-math.floor(x)
            x3=1+math.floor(x)-x
            x4=2+math.floor(x)-x
            y1=1+y-math.floor(y)
            y2=y-math.floor(y)
            y3=1+math.floor(y)-y
            y4=2+math.floor(y)-y
            if x>0 and y>0 and x<X and y<Y and y+y4<Y and x+x4<X:
                mat_x=np.matrix([[K(x1,a),K(x2,a),K(x3,a),K(x4,a)]])
                mat_y=np.matrix(
                [[K(y1,a)],[K(y2,a)],[K(y3,a)],[K(y4,a)]])
                mat_m = np.matrix([[Arpic[int(y-y1), int(x-x1)],
                                    Arpic[int(y-y2), int(x-x1)],
                                    Arpic[int(y+y3), int(x-x1)],
                                    Arpic[int(y+y4), int(x-x1),]],
                                   [Arpic[int(y-y1), int(x-x2),],
                                    Arpic[int(y-y2), int(x-x2),],
                                    Arpic[int(y+y3), int(x-x2),],
                                    Arpic[int(y+y4), int(x-x2),]],
                                   [Arpic[int(y-y1), int(x+x3),],
                                    Arpic[int(y-y2), int(x+x3)],
                                    Arpic[int(y+y3), int(x+x3)],
                                    Arpic[int(y+y4), int(x+x3)]],
                                   [Arpic[int(y-y1), int(x+x4)],
                                    Arpic[int(y-y2), int(x+x4)],
                                    Arpic[int(y+y3), int(x+x4)],
                                    Arpic[int(y+y4), int(x+x4),]]])
                Newpic[j,i]=np.dot(np.dot(mat_x,mat_m),mat_y)
    return Newpic

def K(s, a):
    if (abs(s) >= 0) & (abs(s) <= 1):
        return (a+2)*(abs(s)**3)-(a+3)*(abs(s)**2)+1
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a*(abs(s)**3)-(5*a)*(abs(s)**2)+(8*a)*abs(s)-4*a
    return 0
def MSE(before,after):
    return(np.sqrt((before-after)**2)).mean()
def MSA(before,after):
    return(abs(before-after)).sum()

pic=cv2.imread("images/OrginalZad3.bmp",0)
X,Y=pic.shape
A=NNscale(pic,1)
B=Bilinearscale(pic,1)
C=BicubicScale(pic,1)
D=NNRotation(pic,360)
E=BilinearRotation(pic,360)
F=BicubicRotation(pic,360)
