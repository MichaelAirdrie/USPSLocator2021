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
    imgForEdge = cv.imread('mailbox1.jpg', cv.IMREAD_GRAYSCALE)
    cv.imshow("Origanl", imgForEdge)    
    cv.waitKey(0)
    EdgeDetection(imgForEdge)
def EdgeDetection(imgForEdge):
    newImg = imgForEdge.copy()
    width = imgForEdge.shape[0]
    height = imgForEdge.shape[1]
    imgForEdge = smooth(imgForEdge, 7)
    cv.imshow("Origanl", imgForEdge)
    cv.waitKey(0)
    sigma = 1
    for i in range(0, width - 1):
        for j in range (0, height - 1):
            if (i < width):
                if ((int(imgForEdge.real[i + 1][j]) - int(imgForEdge.real[i][j]) > sigma)or (int(imgForEdge.real[i + 1][j]) - int(imgForEdge.real[i][j]) > sigma)):
                    newImg.real[i][j] = 0
                else:
                    newImg.real[i][j] = 255
            if (j < height):
                if ((int(imgForEdge.real[i][j + 1]) - int((imgForEdge.real[i][j])) > sigma)or ((int(imgForEdge.real[i][j - 1])) - int((imgForEdge.real[i][j])) > sigma)):
                        newImg.real[i][j] = 0
                else:
                    newImg.real[i][j] = 255
    cv.imshow("Origanl", newImg)
    cv.waitKey(0)
def smooth(img, amount):
    width = img.shape[0]
    height = img.shape[1]
    img2 = img.copy()
    for i in range(0, width):
        for j in range(0, height):
            #Go through every pixel in the image
            val = 0            
            if (i < amount) and (j < amount): #if the pixel is too close to a wall and cannot utilize all (amount*2+1) pixels
                for x in range(0, i - amount):
                    for y in range(0, j - amount):
                        val = val + img.real[i - (i - amount) + x][j - (j - amount) + y]
                img2.real[i, j] = int(val/((i - (i - amount))*(j - (j - amount))))
            elif (i < amount) and (j + amount > height): #if the pixel is too close to a wall and cannot utilize all (amount*2+1) pixels
                for x in range(0, i - amount):
                    for y in range(0, j - (j - amount)):
                        val = val + img.real[i - (i - amount) + x][j - (j - amount) + y]
                img2.real[i, j] = int(val/((i - (i - amount))*(j - (j - amount))))
            elif (i + amount > width) and (j + amount > height): #if the pixel is too close to a wall and cannot utilize all (amount*2+1) pixels
                for x in range(0, i - (i - amount)):
                    for y in range(0, j - (j - amount)):
                        val = val + img.real[i - (i - amount) + x][j - (j - amount) + y] 
                img2.real[i, j] = int(val/((i - (i - amount))*(j - (j - amount))))
            elif (i + amount > width) and (j < amount): #if the pixel is too close to a wall and cannot utilize all (amount*2+1) pixels
                for x in range(0, i - (i - amount)):
                    for y in range(0, j - (j - amount)):
                        val = val + img.real[i - (i - amount) + x][j - (j - amount) + y]     
                img2.real[i, j] = int(val/((i - (i - amount))*(j - (j - amount))))
            elif (j < amount):
                for x in range(0, amount):
                    for y in range(0, j - amount):
                        val = val + img.real[i - amount + x][j - (j - amount) + y]  
                img2.real[i, j] = int(val/((j - (j - amount))*(amount)))
            elif (i < amount):
                for x in range(0, i - amount):
                    for y in range(0, amount):
                        val = val + img.real[i - (i - amount) + x][j - amount + y]   
                img2.real[i, j] = int(val/((i - (i - amount))*(amount)))      
            elif (j + amount > height):  
                for x in range(0, amount):
                    for y in range(0, j - (j - amount)):
                        val = val + img.real[i - amount + x][j - (j - amount) + y]                       
                img2.real[i, j] = int(val/((j - (j - amount))*(amount)))
            elif (i + amount > width): 
                for x in range(0, i - (i - amount)):
                    for y in range(0, amount):
                        val = val + img.real[i - (i - amount) + x][j - amount + y]      
                img2.real[i, j] = val/((i - (i - amount))*(amount))                        
            else:
                for x in range(0, amount):
                    for y in range(0, amount):
                        val = val + img.real[i - amount + x][j - amount + y]
                img2.real[i, j] = val/(amount*amount) 
    return img2
main()
