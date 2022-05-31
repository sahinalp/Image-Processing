"""
author: Şahin Alp Akosman
"""

import numpy as np
import warnings
import cv2
from math import*
import skimage.morphology


def reflection(nonReflectedImage,sel_ref):
    """
    This function takes the non-reflected image and the reflection selection and returns the reflection of the non-reflected image.

    nonReflectedImage: the image to be reflected
    sel_ref:
    1-Reflection for x axis
    2-Reflection for y axis
    3-Reflection for x and y axis
    return: reflected image
    """
    [row_size,col_size,colr_size]=nonReflectedImage.shape
    reflectedImage=nonReflectedImage.copy()       
    if sel_ref==1:
        for i in range(0,row_size):
            for j in range(0,col_size):
                for k in range(0,colr_size):
                    reflectedImage[row_size-i-1][j][k]=nonReflectedImage[i][j][k]
                    
    elif sel_ref==2:
        for i in range(0,row_size):
            for j in range(0,col_size):
                for k in range(0,colr_size):
                    reflectedImage[i][col_size-j-1][k]=nonReflectedImage[i][j][k]
    elif sel_ref==3:
        for i in range(0,row_size):
            for j in range(0,col_size):
                for k in range(0,colr_size):
                    reflectedImage[row_size-i-1][col_size-j-1][k]=nonReflectedImage[i][j][k]   
    return reflectedImage

def resize(nonResizedImage,resizeRatio):
    """
    This function takes the non-resized image and the resize ratio and returns the resized image.
    
    nonResizedImage: the image to be resized
    resizeRatio: the ratio must be between 0 and 1
    return: resized image
    """
    if resizeRatio<0 or resizeRatio>1:
        warnings.warn("The resize ratio must be between 0 and 1")
        return nonResizedImage
    [row_size,col_size,colr_size]=nonResizedImage.shape
    rat=1/resizeRatio
    
    new_row_size=int(row_size*resizeRatio)
    new_col_size=int(col_size*resizeRatio)
    a=(new_row_size,new_col_size,3)
    resizedImage=np.zeros(a,dtype=np.uint8)
    row=-1
    col=-1
    act_i=0
    act_j=0
    for i in range(0,new_row_size):
        act_i=act_i+rat
        act_j=0
        row=row+1
        col=-1
        for j in range(0,new_col_size):
            act_j=act_j+rat
            col=col+1
            for k in range(0,3):
                new_i=round(act_i)
                new_j=round(act_j)
                if i>0:
                    old_i=round(act_i-rat)
                else:
                    old_i=0
                if j>0:
                    old_j=round(act_j-rat)
                else:
                    old_j=0
                rat1=new_i-old_i
                rat2=new_j-old_j
                if col_size>=new_j and row_size>=new_i:
                    #Find summation
                    sumOfPixels=0
                    for s1 in range(old_i,new_i):
                        for s2 in range(old_j,new_j):
                            sumOfPixels=sumOfPixels+(nonResizedImage[s1][s2][k])
                    #Take average
                    val=(sumOfPixels/(rat1*rat2))
                    resizedImage[row][col][k]=np.uint8(val)
    return resizedImage

def crop(nonCroppedImage):
    """
    This function takes the non-cropped image and returns the cropped image. 
    To crop please select two diagonel points...

    nonCroppedImage: the image to be cropped
    return: cropped image
    """

    def click_event(event, x, y,flags, params): 
        # if the left button of mouse 
        # is clicked then this 
        # condition executes 
        if len(points)<2 and event == cv2.EVENT_LBUTTONDOWN: 

            # appending the points we 
            # clicked to list 
            points.append((x,y)) 
                        	
            # marking the point with a circle 
            # of center at that point and 
            # small radius 
            cv2.circle(img,(x,y), 4,(0, 0, 0), -1) 
             
            # condition executes 
            # displays the image 
        cv2.imshow('PLEASE SELECT TWO POINTS', img)
        if len(points)==2:
            cv2.destroyWindow('PLEASE SELECT TWO POINTS')
            return points
    img=nonCroppedImage.copy()

    # declare a list to append all the 
    # points on the image we clicked 
    points = [] 
    # show the image 
    cv2.imshow('PLEASE SELECT TWO POINTS',img) 
    # setting mouse call back 
    cv2.setMouseCallback('PLEASE SELECT TWO POINTS', click_event) 
    cv2.waitKey(0)
    [row_size,col_size,colr_size]=nonCroppedImage.shape
    row=-1
    col=-1
    p1=points[0]
    p2=points[1]
    x1=p1[0]
    y1=p1[1]
    x2=p2[0]
    y2=p2[1]
    a=(abs(y1-y2),abs(x1-x2),3)
    croppedImage=np.zeros(a,dtype=np.uint8)
    for i in range(y1,y2):
        row=row+1
        col=-1
        for j in range(x1,x2):
            col=col+1
            for k in range(0,3):
                croppedImage[row][col][k]=nonCroppedImage[i][j][k]
    return croppedImage

def shift(nonShiftedImage,n1,n2):
    """
    This function takes the non-shifted image and the shift values and returns the shifted image.

    nonShiftedImage: the image to be shifted
    n1: shift value for x axis
    n2: shift value for y axis
    return: shifted image
    """

    [row_size,col_size,colr_size]=nonShiftedImage.shape
    row=0
    col=0
                    
    shiftedImage=np.zeros((row_size,col_size,colr_size),dtype=np.uint8)
    for i in range(n1,row_size):
        for j in range(n2,col_size):
            for k in range(0,colr_size):
                shiftedImage[row][col][k]=nonShiftedImage[i-n1][j-n2][k]
            col=1+col
        col=0           
        row=row+1    
    return shiftedImage

def convert(nonConvertedImage,sel_con):
    """
    This function takes the non-converted image and the convertion type and returns the converted image.
    nonConvertedImage: the image to be converted
    sel_con:
    1-RGB to HSI
    2-RGB to HSV
    3-RGB to YIQ
    4-HSI to RGB
    5-HSV to RGB
    6-YIQ to RGB
    return: converted image
    """
    #size
    [row_size,col_size,colr_size]=nonConvertedImage.shape
    #im2double
    convertedImage=np.zeros((row_size,col_size,colr_size),dtype=np.float64)
    for i in range(0,row_size):
        for j in range(0,col_size):
            for k in range(0,colr_size):
                convertedImage[i][j][k]=nonConvertedImage[i][j][k]/255
    nonConvertedImage=convertedImage.copy()
    if sel_con==1:
        for i in range(0,row_size):
           for j in range(0,col_size):
                R=nonConvertedImage[i][j][2]
                G=nonConvertedImage[i][j][1]
                B=nonConvertedImage[i][j][0]
                #H
                if R==G and G==B:
                    H=0
                elif R>G and R>B:
                    H=60*(G-B)/(R-min(G,B))
                elif G>R and G>B:
                    H=60*(B-R)/(G-min(R,B))+120
                elif B>R and B>G:
                    H=60*(R-G)/(B-min(R,G))+240
                #S
                if R==G and G==B:
                    S=0
                else:
                    S=1-(min(R,G,B)/max(R,G,B))
                #I
                I=(R+G+B)/3
                convertedImage[i][j][0]=H
                convertedImage[i][j][1]=S
                convertedImage[i][j][2]=I 
    elif sel_con==2:
        for i in range(0,row_size):
           for j in range(0,col_size):
                R=nonConvertedImage[i][j][2]
                G=nonConvertedImage[i][j][1]
                B=nonConvertedImage[i][j][0]
                #H
                if R==G and G==B:
                    H=0
                elif R>G and R>B:
                    H=60*(G-B)/(R-min(G,B))
                elif G>R and G>B:
                    H=60*(B-R)/(G-min(R,B))+120
                elif B>R and B>G:
                    H=60*(R-G)/(B-min(R,G))+240
                #S
                if R==G and G==B:
                    S=0
                else:
                    S=1-(min(R,G,B)/max(R,G,B))
                #V
                V=max(R,G,B)
                convertedImage[i][j][0]=H
                convertedImage[i][j][1]=S
                convertedImage[i][j][2]=V

    elif sel_con==3:
        for i in range(0,row_size):
           for j in range(0,col_size):
                R=nonConvertedImage[i][j][2]
                G=nonConvertedImage[i][j][1]
                B=nonConvertedImage[i][j][0]                
                Y=(0.2989*R)+(0.5870*G)+(0.1140*B)
                I=(0.596*R)-(0.274*G)-(0.322*B)
                Q=(0.211*R)-(0.523*G)+(0.312*B)            
                convertedImage[i][j][2]=Y
                convertedImage[i][j][1]=I
                convertedImage[i][j][0]=Q
    elif sel_con==4:
        for i in range(0,row_size):
           for j in range(0,col_size):
                #H(hue) S(saturation) I 
                H=nonConvertedImage[i][j][2]
                S=nonConvertedImage[i][j][1]
                I=nonConvertedImage[i][j][0]                
                if H>=0 and H<120:
                    B=I*(1-S)
                    R=I*(1+S*cos(H)/cos(60-H))
                    G=1-(R+B)
                elif H>=120 and H<240:
                    H=H-120
                    R=I*(1-S)
                    G=I*(1+S*cos(H)/cos(60-H))
                    B=1-(R+G)
                elif H>=240 and H<=360:
                    H=H-240
                    G=I*(1-S)
                    B=I*(1+S*cos(H)/cos(60-H))
                    R=1-(G+B)   
                convertedImage[i][j][2]=R
                convertedImage[i][j][1]=G
                convertedImage[i][j][0]=B    
    elif sel_con==5:
        for i in range(0,row_size):
           for j in range(0,col_size):
                #H(hue) S(saturation) V(value)
                H=nonConvertedImage[i][j][2]
                S=nonConvertedImage[i][j][1]
                V=nonConvertedImage[i][j][0]
                #chroma
                c=V*S
                #the RGB component with the smallest value
                m=V-c
                x=c*(1-abs((H/60)%2-1))
                if 0<=H and H<60:
                    convertedImage[i][j][2]=c+m
                    convertedImage[i][j][1]=x+m
                    convertedImage[i][j][0]=m
                elif 60<=H and H<120:
                    convertedImage[i][j][2]=x+m
                    convertedImage[i][j][1]=c+m
                    convertedImage[i][j][0]=m
                elif 120<=H and H<180:
                    convertedImage[i][j][2]=m
                    convertedImage[i][j][1]=c+m
                    convertedImage[i][j][0]=x+m
                elif 180<=H and H<240:
                    convertedImage[i][j][2]=m
                    convertedImage[i][j][1]=x+m
                    convertedImage[i][j][0]=c+m
                elif 240<=H and H<300:
                    convertedImage[i][j][2]=x+m
                    convertedImage[i][j][1]=m
                    convertedImage[i][j][0]=c+m
                elif 300<=H and H<360:
                    convertedImage[i][j][2]=c+m
                    convertedImage[i][j][1]=m
                    convertedImage[i][j][0]=x+m
                else:
                    convertedImage[i][j][2]=m
                    convertedImage[i][j][1]=m
                    convertedImage[i][j][0]=m
    elif sel_con==6:
        for i in range(0,row_size):
           for j in range(0,col_size):
                Y=nonConvertedImage[i][j][2]
                I=nonConvertedImage[i][j][1]
                Q=nonConvertedImage[i][j][0]
                convertedImage[i][j][2]=Y+0.956*I+0.621*Q
                convertedImage[i][j][1]=Y-0.272*I-0.647*Q
                convertedImage[i][j][0]=Y-1.106*I+1.703*Q
    return convertedImage

def hist(orginalImage):
    """
    This method is used to calculate the histogram of the image.
    orginalImage: The image which is to be converted.
    return: The histogram of the image.
    """
    
    [row_size,col_size,clr_size]=orginalImage.shape
    min_image=np.min(orginalImage)
    max_image=np.max(orginalImage)
    a=(row_size,col_size,clr_size)
    strechedImage=np.zeros(a,dtype=np.uint8)
    for i in range(0,row_size):
        for j in range(0,col_size):
            strechedImage[i][j][0]=((255/(max_image-min_image))*(orginalImage[i][j][0]-min_image))
            strechedImage[i][j][1]=((255/(max_image-min_image))*(orginalImage[i][j][1]-min_image))
            strechedImage[i][j][2]=((255/(max_image-min_image))*(orginalImage[i][j][2]-min_image))
    return strechedImage

def hist_eq(orginalImage):
    """
    This method is used to get histogram of image
    orginalImage:orginal image
    return:histogram of image
    """
    hist=np.zeros(256)
    try:
        [row_size,col_size,clr_size]=orginalImage.shape
        for i in range(0,row_size):
            for j in range(0,col_size):
                for k in range(0,clr_size):
                    hist[orginalImage[i][j][k]]+=1 
    except:
        [row_size,col_size]=orginalImage.shape
        for i in range(0,row_size):
            for j in range(0,col_size):
                hist[orginalImage[i][j]]+=1
    finally:
        return hist

def hieq(image):
    """
    This method is used to equalize the histogram of the image.
    image: image to be equalized
    returns: equalized image
    """
    [row_size,col_size,clr_size]=image.shape
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    #Getting histogram of image
    hisOfImage=hist_eq(image)
    
             
    #Genereating pdf 
    pdf_image=[]
    for i in range(0,256):
        pdf_image.append(0)
    for i in range(0,256):
        pdf_image[i]=(1/(row_size*col_size))*hisOfImage[i]
    
    #Generaiting cdf
    cdf_image=[]
    for i in range(0,256):
        cdf_image.append(0)
        
    cdf_image[0]=pdf_image[0]
    
    for i in range(1,256):
       cdf_image[i]=cdf_image[i-1]+pdf_image[i]
    
    for i in range(0,256):
        cdf_image[i]=round(cdf_image[i]*255)
    
    outputImage=np.zeros((row_size,col_size),dtype=np.uint8)
    #Output image
    for i in range(0,row_size):
       for j in range(0,col_size):
          k=(image[i][j]+1)
          outputImage[i][j]=cdf_image[k-1]
             
    return outputImage

def wh_bl(image,black_level,white_level):
    """
    This method is used to convert the image to black and white.

    image: image to be converted
    black_level and white_level are the black and white levels of the image.
    returns: black and white image
    """
    [row_size,col_size,clr_size]=image.shape
    
    outImage=np.zeros((row_size,col_size,clr_size),dtype=np.uint8)
    
    for i in range(0,row_size):
       for j in range(0,col_size):
           for k in range(0,3):
                if image[i,j,k]<black_level:
                   outImage[i,j,k]=0
                elif image[i,j,k]>white_level:
                    outImage[i,j,k]=255
                else:
                    outImage[i,j,k]=image[i,j,k]

    return outImage

def insertionsort(A):
    """
    This function is used to sort the array in ascending order.
    param: A is the array to be sorted.
    return: A is the sorted array.
    """

    for i in range(1, len(A)):
        key = A[i]
        j = i - 1
        while j >= 0 and key < A[j]:
            A[j + 1] = A[j]
            j = j - 1
        A[j + 1] = key
    return A

def medFilter(image,num):
    """
    This is a median filter.
    It is used to reduce the noise in the image. 
    
    image: image to be filtered
    num: The size of the filter
    return: filtered image
    """
    if len(image.shape)==3:
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    [row_size,col_size]=image.shape
    a=(row_size,col_size)
    
    resultImage=np.zeros(a,dtype=np.uint8)
    if num==2 or num==3:
        newImage=np.pad(image,(1,1),'constant')
    elif num==4:
        newImage=np.pad(image,(1,1),'constant')
        newImage=np.pad(newImage,(0,1),'constant')
    else:
        if num%2==0:
            newImage=np.pad(image,(int((num/2)-1),int((num/2)-1)),'constant')
            newImage=np.pad(newImage,(0,1),'constant')
        else:
            newImage=np.pad(image,(int((num-1)/2),int((num-1)/2)),'constant')
        
    for i in range(0,row_size):
       for j in range(0,col_size):
           
           unsortedList=[]
           for x in range(i,i+num):
               for y in range(j,j+num):
                   unsortedList.append(newImage[x][y])
           sortedList=insertionsort(unsortedList)           
           if (num*num%2)==1:
               val=int(((num*num)-1)/2)
               resultImage[i][j]=sortedList[1+val]
           else:
               val=int((num*num)/2)
               resultImage[i][j]=int((sortedList[val])/2+(sortedList[val+1])/2)
    
    resultImage=np.uint8(resultImage)
    return resultImage

def minFilter(image,order):
    """
    This is a min filter.
    It is used to reduce the noise in the image.
    
    image: image to be filtered
    order: The size of the filter
    return: filtered image
    """
    if len(image.shape)==3:
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    [row_size,col_size]=image.shape
    a=(row_size,col_size)
    
    resultImage=np.zeros(a,dtype=np.uint8)
    num=3
    newImage=np.pad(image,(1,1),'constant')
    
    for i in range(0,row_size):
       for j in range(0,col_size):
           unsortedList=[]
           for x in range(i,i+num):
               for y in range(j,j+num):
                   unsortedList.append(newImage[x][y])
                   
           sortedList=insertionsort(unsortedList)
           
           resultImage[i][j]=sortedList[order]
           
    resultImage=np.uint8(resultImage)
    return resultImage

def maxFilter(image):
    """
    This is a max filter.
    It is used to reduce the noise in the image.

    image: image to be filtered
    return: filtered image
    """
    if len(image.shape)==3:
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    [row_size,col_size]=image.shape
    a=(row_size,col_size)
    
    resultImage=np.zeros(a,dtype=np.uint8)
    num=3
    newImage=np.pad(image,(1,1),'constant')
    
    for i in range(0,row_size):
       for j in range(0,col_size):
           unsortedList=[]
           for x in range(i,i+num):
               for y in range(j,j+num):
                   unsortedList.append(newImage[x][y])
                   
           sortedList=insertionsort(unsortedList)
           
           resultImage[i][j]=sortedList[8]
           
    resultImage=np.uint8(resultImage)

    return resultImage

def thresholdFilter(image):
    """
    This is a threshold filter.
    It is used to reduce the noise in the image.

    image: image to be filtered
    return: filtered image
    """
    if len(image.shape)==3:
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    [row_size,col_size]=image.shape
    a=(row_size,col_size)

    threshold=np.mean(image)
    outputImage=np.zeros(a,dtype=np.double)
    for i in range(0,row_size):
       for j in range(0,col_size):
           if image[i,j]<=threshold:
              outputImage[i,j]=0
           elif image[i,j]>threshold:
               outputImage[i,j]=1
    outputImage=np.uint8(outputImage)
    return outputImage

def highpassFilter(image):
    """
    This is a highpass filter.
    It is used to reduce the noise in the image.

    image: image to be filtered
    return: filtered image
    """
    if len(image.shape)==3:
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    [row_size,col_size]=image.shape
    a=(row_size,col_size)

    outputImage=np.zeros(a,dtype=np.double)
    HighpassKernel = [[-1/9,-1/9,-1/9],[-1/9,8/9,-1/9],[-1/9,-1/9,-1/9]]
    for i in range(0,row_size):
         for j in range(0,col_size):
              if i==0 or i==row_size-1 or j==0 or j==col_size-1:
                outputImage[i,j]=0
              else:
                outputImage[i,j]=image[i-1:i+2,j-1:j+2]*HighpassKernel
    
    outputImage=np.uint8(outputImage)
    return outputImage

def inverseFilter(image):
    """
    This is a inverse filter.
    It is used to reduce the noise in the image.

    image: image to be filtered
    return: filtered image
    """
    if len(image.shape)==3:
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    [row_size,col_size]=image.shape
    a=(row_size,col_size)

    H=zeros(a,dtype=complex)
    inv_filter=zeros(a,dtype=complex)
    outputImage=zeros(a,dtype=np.double)

    for i in range(0,row_size):
        for j in range(0,col_size):
            H[i,j]=(T*sin(pi*(i*a+j*b))*np.exp(-1j*pi*(i*a+j*b)))/(pi*(i*a+j*b))

    for i in range(0,row_size):
        for j in range(0,col_size):
            A=1/H[i,j]
            inv_filter[i,j]=A*image[i,j]

    outputImage=uint8(real(inv_filter))
    return outputImage

def wienerFilter(image):
    """
    This is a wiener filter.
    It is used to reduce the noise in the image.

    image: image to be filtered
    return: filtered image
    """
    if len(image.shape)==3:
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    [row_size,col_size]=image.shape
    a=(row_size,col_size)

    H=zeros(a,dtype=complex)
    Hinv=zeros(a,dtype=complex)
    Hmultiply=zeros(a,dtype=complex)
    wie_filter=zeros(a,dtype=complex)
    outputImage=zeros(a,dtype=np.double)

    for i in range(0,row_size):
        for j in range(0,col_size):
            H[i,j]=(T*sin(pi*(i*a+j*b))*np.exp(-1j*pi*(i*a+j*b)))/(pi*(i*a+j*b))
            
    #take conj of H
    for i in range(0,row_size):
        for j in range(0,col_size):
            Hinv[i,j]=conj(H[i,j])
            
    #multiply H and conj of H
    for i in range(0,row_size):
        for j in range(0,col_size):
            Hmultiply[i,j]=conj(H[i,j])*Hinv[i,j]
    #wiener filter
    for i in range(0,row_size):
        for j in range(0,col_size):
            A=Hmultiply[i,j]/(H[i,j]*(Hmultiply[i,j]+K))
            wie_filter[i,j]=A*image[i,j]

    outputImage=uint8(real(wie_filter))
    return outputImage

def Erosion(image,n,I):
    """
    This is erosion function. It is used to erode away the boundaries of regions
    
    image: image to be filtered
    n: size of the kernel
    I: intensity of the kernel
    return: filtered image
    """
    [row_size,col_size]=image.shape
    outputImage= np.zeros((image.shape[0],image.shape[1]), dtype=np.uint8)
    if n<3 and n>0:
        c=1
        m=3
    elif n>=3:
        c=n-1
        m=n*2-1
    padArrayImage=np.pad(image,c,'constant')  
    SE=skimage.morphology.disk(c, dtype=np.int32)
    [N,M]=SE.shape
    K=[]
    for i in range(0,N):
        for j in range (0,M):
            if SE[i,j]==1:
                K.append([i,j])
    X=np.array(K,dtype=np.uint8)
    for i in range(0,row_size):
        for j in range(0,col_size):
            temp= padArrayImage[i:i+m, j:j+m]
            product= temp*SE
            sum_product=0
            for x in K:
                if (product[x[0],x[1]])==0:
                    sum_product=sum_product+1
            if sum_product==0:
                outputImage[i,j]=I[i,j]
            else:
                outputImage[i,j]=0   
    return outputImage

def Dilation(image,n,I):
    """
    This is a dilation function. It is used to dilate the boundaries of regions
    

    image: image to be filtered
    n: size of the kernel
    I: intensity of the kernel
    return: filtered image
    """
    [row_size,col_size]=image.shape
    outputImage= np.zeros((image.shape[0],image.shape[1]), dtype=np.uint8)
    if n<3 and n>0:
        c=1
        m=3
    elif n>=3:
        c=n-1
        m=n*2-1
    padArrayImage=np.pad(image,c,'constant')
    SE=skimage.morphology.disk(c, dtype=np.int32)
    [N,M]=SE.shape
    K=[]
    for i in range(0,N):
        for j in range (0,M):
            if SE[i,j]==1:
                K.append([i,j])
    X=np.array(K,dtype=np.uint8)          
    for i in range(0,row_size):
        for j in range(0,col_size):
            temp= padArrayImage[i:i+m, j:j+m]
            product= temp*SE
            sum_product=sum(product)
            if sum_product>0:
                outputImage[i,j]=I[i,j]
            else:
                outputImage[i,j]=0
            
    return outputImage

def otsu(image):
    """
    This is a otsu thresholding function. It is used to threshold the image.
    
    image: image to be filtered
    return: filtered image
    """
    [row,col]=image.shape
    H = []
    for i in range(0,256):
        H.append(0)
        
    for i in range(0,row):
        for j in range(0,col):
            H[image[i,j]] = H[image[i,j]] + 1
             
    #Generating PDF out of Histogram by diving by total no. of pixels
    pdf_image = []
    for i in range(0,256):
        pdf_image.append(0)
    for i in range (0,256)   :
        pdf_image[i]=(1/(row*col))*H[i]
    A=[]
    for i in range (0,256):
        A.append(i)
    mean=[]
    for i in range(0,256):
        mean.append(0)
    for i in range(0,256):
        mean[i]=pdf_image[i]*A[i]
    probability1=[]
    for i in range(0,256):
        probability1.append(0)
    probability2=[]
    for i in range(0,256):
        probability2.append(0)
    mean1=[]
    for i in range(0,256):
        mean1.append(0)
    mean2=[]
    for i in range(0,256):
        mean2.append(0)
    variance=[]
    for i in range(0,256):
        variance.append(0)
    min_value=0
    max_value=255
    for i in range(min_value,max_value):
        probability1[i]=np.sum(pdf_image[min_value:i])
        probability2[i]=1-probability1[i]
        #mean of class A
        mean1[i]=np.sum(mean[min_value:i])/probability1[i]
        #mean of class B
        mean2[i]=np.sum(mean[i+1:max_value])/probability2[i]
        #global mean
        mg=probability1[i]*mean1[i]+probability2[i]*mean2[i]
        #between class variance
        variance[i]=probability1[i]*np.power((mean1[i]-mg),2)+probability2[i]*np.power((mean2[i]-mg),2)
    
    #finding index number of max_value value of variance
    thnum=0
    tmp=0
    for i in range(min_value,max_value):
        if variance[i]>tmp and variance[i]!=nan:
            tmp=variance[i]
            thnum=i
    #Thresholding/Binarization
    outputImage= np.zeros((row,col), dtype=np.uint8)
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            if image[i,j]<=thnum:
                outputImage[i,j]=0
            else:
                outputImage[i,j]=255
    return outputImage

def regiongrowing(image):
    """
    This is a region growing function. It is used to grow the region.

    image: image to find the region
    return: image
    """
    #finding the threshold value
    [row,col]=image.shape
    temp=[]
    for i in range(0,row):
        for j in range(0,col):
            temp.append(image[i,j])
    thfactor=statistics.mode(temp)
    #finding the lowest pixel of the image
    y1=0
    x1=0
    k=0
    for i in range(0,row):
        for j in range(0,col):
            if k<image[i,j]:
                k=image[i,j]
                x1=i
                y1=j
    
    def regiongrowingfunction(image,A,B,thfactor):
            """
            image: image to be filtered
            A: x coordinate of the seed point
            B: y coordinate of the seed point
            thfactor: threshold factor
            return: image
            """
            if image[A,B]>thfactor and image[A,B]!=255:
                image[A,B]=255
                image=regiongrowingfunction(image, A+1, B, thfactor)
                image=regiongrowingfunction(image, A, B+1, thfactor)
                image=regiongrowingfunction(image, A-1, B, thfactor)
                image=regiongrowingfunction(image, A, B-1, thfactor)
                
                image=regiongrowingfunction(image, A+1, B+1, thfactor)
                image=regiongrowingfunction(image, A+1, B-1, thfactor)
                image=regiongrowingfunction(image, A-1, B-1, thfactor)
                image=regiongrowingfunction(image, A-1, B+1, thfactor)
                
            return image

    outputImage=regiongrowingfunction(image, x1, y1, thfactor)
    return outputImage
