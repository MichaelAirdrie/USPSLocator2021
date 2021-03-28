from __future__ import print_function
import cv2 as cv
import numpy as np
from math import sqrt
import os
path = '../mailboxes/'
def akaze(file):
    img1 = cv.imread(path + file, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(path + 'sticker.jpg', cv.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        print('Could not open or find the images!')
        exit(0)
    akaze = cv.AKAZE_create()
    kpts1, desc1 = akaze.detectAndCompute(img1, None)
    kpts2, desc2 = akaze.detectAndCompute(img2, None)
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)
    nn_matches = matcher.knnMatch(desc1, desc2, 2)
    matched1 = []
    matched2 = []
    nn_match_ratio = 0.8 # Nearest neighbor matching ratio
    for m, n in nn_matches:
        if m.distance < nn_match_ratio * n.distance:
            matched1.append(kpts1[m.queryIdx])
            matched2.append(kpts2[m.trainIdx])
    src_pts = np.float32([ m.pt for m in matched1 ]).reshape(-1,1,2)
    dst_pts = np.float32([ m.pt for m in matched2 ]).reshape(-1,1,2)
    if (min(src_pts.size, dst_pts.size) < 7):
        print("not enough matches for " + file)
        return 0
    homography, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    inliers1 = []
    inliers2 = []
    good_matches = []
    inlier_threshold = 2.5 # Distance threshold to identify inliers with homography check
    for i, m in enumerate(matched1):
        col = np.ones((3,1), dtype=np.float64)
        col[0:2,0] = m.pt
        col = np.dot(homography, col)
        col /= col[2,0]
        dist = sqrt(pow(col[0,0] - matched2[i].pt[0], 2) +\
                    pow(col[1,0] - matched2[i].pt[1], 2))
        if dist < inlier_threshold:
            good_matches.append(cv.DMatch(len(inliers1), len(inliers2), 0))
            inliers1.append(matched1[i])
            inliers2.append(matched2[i])
    res = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(img1, inliers1, img2, inliers2, good_matches, res)
    inlier_ratio = len(inliers1) / float(len(matched1))
    print('A-KAZE Matching Results')
    print('*******************************')
    print('# Keypoints 1:                        \t', len(kpts1))
    print('# Keypoints 2:                        \t', len(kpts2))
    print('# Matches:                            \t', len(matched1))
    print('# Inliers:                            \t', len(inliers1))
    print('# Inliers Ratio:                      \t', inlier_ratio)
    cv.imwrite(('stickerChecked\\' + file), res)
for file in os.listdir(path): #loop through each mailbox and take a look at keypoints
    if (file != 'sticker.jpg'):
        print(file)
        akaze(file)
