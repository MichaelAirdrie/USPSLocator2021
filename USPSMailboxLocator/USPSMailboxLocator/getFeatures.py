import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
path = 'mailboxes/' #directory where mailbox pictures are stored
def getKeypoints(path):
    window = 'Mailbox :)'
    src = cv.imread(path, 0)
    #cv.imshow(window, src)
    #cv.waitKey()
    descriptor = cv.SIFT_create()
    kp, des = descriptor.detectAndCompute(src, None)
    keyPointImage = src.copy()
    keyPointImage = cv.drawKeypoints(src, kp, keyPointImage, (255, 0, 0))
    #cv.imshow(window, keyPointImage)
    #cv.waitKey()
    
for file in os.listdir(path): #loop through each mailbox and take a look at keypoints
    if (file != 'sticker.jpg'):
        pathToImage = path + file
        getKeypoints(pathToImage)
        print(file)


#I'm thinking we get a picture of a USPS sticker and see if you can look through
#all of the mailbox images and see if we if we can use the presence of the sticker as an indicator
#that it is indeed a USPS mailbox
