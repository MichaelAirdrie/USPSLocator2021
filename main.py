import matplotlib.pyplot as plt
import Akaze, FlannKnnMatch, BFMatcher
import cv2 as cv
import numpy as np
import os
def main(ransac, distance_ratio, akaze_inlier_ratio):
    [AkeyPointNumMailbox, AkeyPointNumSticker, AmatchesNum, AImageSize, AStickerSize] = Akaze.main(akaze_inlier_ratio)
    [FkeyPointNumMailbox, FkeyPointNumSticker, FmatchesNum, FImageSize, FStickerSize] = FlannKnnMatch.main(distance_ratio)
    [BkeyPointNumMailbox, BkeyPointNumSticker, BmatchesNum, BImageSize, BStickerSize] = BFMatcher.main()
    if (len(AmatchesNum) == 0):
        AmatchesNum.append(1)
    if (len(FmatchesNum) == 0):
        FmatchesNum.append(1)
    if (len(BmatchesNum) == 0):
        BmatchesNum.append(1)
    
    return (sum(FmatchesNum)/len(FmatchesNum)), distance_ratio, (sum(BmatchesNum)/len(BmatchesNum)), ransac, (sum(AmatchesNum)/len(AmatchesNum)), akaze_inlier_ratio
ransac = 5.0
distance_ratio = 0.3
akaze_inlier_ratio = 2.5
AmatchesAvgNum = []
FmatchesAvgNum = []
BmatchesAvgNum = []
distance_ratios = []
ransacs = []
akaze_inlier_ratios = []

for i in range(0,6):
    retArgs = main(ransac, distance_ratio, akaze_inlier_ratio)
    AmatchesAvgNum.append(retArgs[4])
    FmatchesAvgNum.append(retArgs[0])
    BmatchesAvgNum.append(retArgs[2])
    distance_ratios.append(retArgs[1])
    ransacs.append(retArgs[3])
    akaze_inlier_ratios.append(retArgs[5])
    distance_ratio = distance_ratio + 0.1
    akaze_inlier_ratio = akaze_inlier_ratio + 0.5
    
plt.plot(FmatchesAvgNum, distance_ratios,  'v', label='Flann')
plt.ylabel("Distance Ratio")
plt.xlabel("Matches")
plt.title("Matches to Distance Ratio")
plt.legend(loc='best')
plt.savefig(str(distance_ratio)+ " Distance Ratio.png")
plt.clf()
        
plt.plot(BmatchesAvgNum, ransacs, '^', label='Brute Force')
plt.ylabel("Ransac Level")
plt.xlabel("Matches")
plt.title("Matches to Ransac Value")
plt.legend(loc='best')
plt.savefig(str(ransac)+ " Brute Force.png")
plt.clf()
        
plt.plot(AmatchesAvgNum, akaze_inlier_ratios, 'o', label='Akaze')
plt.ylabel("Inlier Ratio")
plt.xlabel("Matches")
plt.title("Matches to Akaze Inlier Ratio")
plt.legend(loc='best')
plt.savefig(str(akaze_inlier_ratio)+ " Akaze Inlier Ratio.png")
plt.clf()
plt.clf()


    

