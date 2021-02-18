import numpy as np 
import cv2 as cv
#-------- OVERVIEW AND IDEAS ---------

#some of the determinents of whether or not a USPS mailbox is within an image are:
#blue,  legs,   curvature along top,    Sticker...
#We want to somehow add weights to each of the above qualities and sum them up
#this summed up value will be an quantitative number where:
#if (sum(blue,  legs,   curvature along top,    Sticker) > threshold):
#   return(True)
#else:
#   return(False)
#for the curvature and legs I would think we should use edge detection and detect certain patterns
#depending on how many of these patterns are present, the score will change accordingly.
#for the blue portion, I would think the color blue by itself is not a good
#indicator of whether or not it is a USPS mailbox. However if between the edges in the edge detection
#there is the color blue then it is much more likely. For example if there is a large amount of blue
#between the edges detected in the edge detection part of the algorithm. (if their is blue within a
#likely shape of a USPS mailbox the odds of it being a USPS mailbox grows more than if it is not the
#shape of a USPS mailbox. this is intuitive because there are alot of things that are blue... but
#there are much fewer things that are blue AND in the shape of a USPS mailbox...

#-------- SIGNIFICANT CHALLANGES ---------

#How do we represent a good model for what the shape of a USPS Mailbox is... There are a ton of angles
#and distance you can be from the box. From a coding standpoint how would we account for this?

#How do we determine whether or not the color blue lies between the shape of the mailbox. How do
#we account for the different shades of blue depending on the lighting and camera that the
#picture was taken?

#Identifying the sticker with varying sizes and angles, Viewing the sticker from head on will be
#way different than viewing it on an angle. the sticker might also mess up the blue detection as it
#lies within the edges of the mailbox (could lower the score by accident) which means that accounting
#for it will be very important

def main():
    img = img = cv.imread('mailbox1.jpg')
    cv.imshow("Origanl", img)    
    cv.waitKey(0)   
main()
