import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
path = 'mailboxes/1.jpg'
def getKeypoints(path):
    window = 'Mailbox :)'
    src = cv.imread(path, 0)
    cv.imshow(window, src)
    cv.waitKey()
    descriptor = cv.SIFT_create()
    kp, des = descriptor.detectAndCompute(src, None)
    keyPointImage = src.copy()
    keyPointImage = cv.drawKeypoints(src, kp, keyPointImage, (255, 0, 0))
    cv.imshow(window, keyPointImage)
    cv.waitKey()
getKeypoints(path)
