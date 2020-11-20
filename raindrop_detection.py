import cv2
import numpy as np
import argparse 
import statistics as st
import os
import timeit

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help=" input image")
args = vars(ap.parse_args())

if not(os.path.isdir('./rain_detection')):
    os.mkdir("./rain_detection")

if not(os.path.isdir('./binary_mask')):
    os.mkdir("./binary_mask")
base = os.path.basename(args["image"])
file_name = os.path.splitext(base)[0]







start = timeit.default_timer()



def mask( img_path ) :

    image = cv2.imread(img_path)
    image = cv2.resize(image, (640,380))


    result = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilate = cv2.dilate(thresh, kernel, iterations=1)
    dilate1 = cv2.dilate(thresh1, kernel, iterations=1)

    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts1 = cv2.findContours(dilate1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts1 = cnts1[0] if len(cnts1) == 2 else cnts1[1]

    ROI_num = 0

    L = []
    L1 = []
    for c in cnts :
        x,y,w,h = cv2.boundingRect(c)
        a = h * w
        L.append(a)

    m = st.median(L)

    maximum = max(L)
    index = L.index(maximum)

    while m*100 < maximum :
        del cnts[index]
        del L[index]
        maximum = max(L)
        index = L.index(maximum)




       
    for c in cnts1 :
        x,y,w,h = cv2.boundingRect(c)
        a = h * w
        L1.append(a)

    m1 = st.median(L1)

    maximum1 = max(L1)
    index1 = L1.index(maximum1)

    while m1*100 < maximum1 :
        del cnts1[index1]
        del L1[index1]
        maximum1 = max(L1)
        index1 = L1.index(maximum1)



    mask = np.zeros([380,640])

    for c in cnts1:
        x,y,w,h = cv2.boundingRect(c)
        ROI = image[y:y+h, x:x+w]
        mask[y:y+h, x:x+w] = 255
        value = cv2.Laplacian(ROI, cv2.CV_64F).var()  
        cv2.rectangle(result, (x, y), (x + w, y + h), (36,255,12), 2)
        #cv2.putText(result, "{0:.2f}".format(value), (x,y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)
        #cv2.imshow("ROI_{}".format(ROI_num), ROI)
        ROI_num += 1
        
        #print('ROI_Number: {}, Value: {}'.format(ROI_num, value))

    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        ROI = image[y:y+h, x:x+w]
        mask[y:y+h, x:x+w] = 255
        value = cv2.Laplacian(ROI, cv2.CV_64F).var()  
        cv2.rectangle(result, (x, y), (x + w, y + h), (36,255,12), 2)
        #cv2.putText(result, "{0:.2f}".format(value), (x,y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)
        #cv2.imshow("ROI_{}".format(ROI_num), ROI)
        ROI_num += 1
        
        #print('ROI_Number: {}, Value: {}'.format(ROI_num, value))

    cv2.imwrite('mask.png',mask)
    
  
                        



    mask1 = cv2.imread('mask.png')
    mask1 = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)
    res = cv2.bitwise_and(dilate1,dilate1,mask = dilate)
    res1 = cv2.bitwise_and(res,res,mask = mask1)
    #print(dilate1.shape)
    #cv2.imwrite('mask.png', mask)
    #cv2.imshow('dilate1', dilate1)
    #cv2.imshow('resultat', res1)
    #cv2.imshow('dilate', dilate)
    cv2.imwrite('./binary_mask/mask_'+base, res1)
    cv2.imwrite('./rain_detection/out_'+base, result)
    #cv2.waitKey(0)
    os.remove('mask.png')


mask(args["image"])    
stop = timeit.default_timer()
print('Time: ', stop - start)  
