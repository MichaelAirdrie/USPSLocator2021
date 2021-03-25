import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
path = 'mailboxes/' #directory where mailbox pictures are stored
def hasSticker(path, sticker):
    window = 'Mailbox :)'
    src = cv.imread(path, 0)
    descriptor = cv.SIFT_create()
    srcKeyPoints, srcDescriptor = descriptor.detectAndCompute(src, None)
    stickerKeyPoints, stickerDescriptor = descriptor.detectAndCompute(sticker, None)
    
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(srcDescriptor,stickerDescriptor,k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)> 3:
        src_pts = np.float32([ srcKeyPoints[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ stickerKeyPoints[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), 11) )
        matchesMask = None
    if len(good)> 10:
        matchesMask = mask.ravel().tolist()
        h,w = src.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        right = cv.polylines(src,[np.int32(dst)],True,255,3, cv.LINE_AA)
        draw_params = dict(matchColor = (0,255,0), singlePointColor = None,matchesMask = matchesMask, flags = 2)
        drawnImage = cv.drawMatches(src,srcKeyPoints,sticker,stickerKeyPoints,good,None,**draw_params)
        cv.imwrite(('stickerChecked\\' + file), drawnImage)
        
    else:
        return ("not a mailbox")
for file in os.listdir(path): #loop through each mailbox and take a look at keypoints
    if (file != 'sticker.jpg'):
        pathToImage = path + file
        x = hasSticker(pathToImage, cv.imread((path + 'sticker.jpg')))
        print(file)


#I'm thinking we get a picture of a USPS sticker and see if you can look through
#all of the mailbox images and see if we if we can use the presence of the sticker as an indicator
#that it is indeed a USPS mailbox
