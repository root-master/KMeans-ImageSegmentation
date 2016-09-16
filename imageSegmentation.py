#import the necessary packages
import sys
sys.path.append('/Users/snehabelkhale/.virtualenvs/cv/lib/python2.7/site-packages')
import cv2
import time
import numpy as np

'''
Function: Segment
Uses the K-Means Clustering Algorithm to segment images based on combinations of HSV or RGB Channels.
Based on the color profile of the image, experimenting with different combinations can produce beautiful and alternative color
posterizing. To enhance this posterizing effect and attain very unorthodox coloring, input "1" for the fourth input parameter

note:Check out the attached powerpoint for some examples ...

Inputs:
A: HSV/RGB image to be segmented
Channel: Tuple argument with corresponding Channel numbers
    Ex. (1,2) -> to segment with the second and third channel
        (1,) -> to only segment according the the second channel
        note: you must provide the channel numbers in chronological order
K: Number of segmentation clusters
Enhance: Input 1 for extra visual effects, 0 for standard

Output:
Segmented Image

'''


def segment(A, Channel, K, Enhance):
    # make sure that Enhance parameter is a 1 or 0
    if Enhance != 1 and Enhance != 0:
        print ('Invalid input for Enhance Parameter, must be 1 or 0')
        return
    # set criteria for k-means algorithm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)

    # if user has selected to cluster by one channel
    if len(Channel) == 1:
        Z = A[:, :, Channel].copy()
        Z = Z.reshape((-1, 1))
        Z = np.float32(Z)
        # obtain centers using kmeans
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # apply the segmentation to the image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((A[:, :, Channel].shape))
        A[:, :, Channel] = res2

    # if user has selected to cluster by two channels
    elif len(Channel) == 2:
        Channel1 = Channel[0]
        Channel2 = Channel[1]
        # pull out the required channels
        Z1 = A[:, :, Channel1].copy()
        Z2 = A[:, :, Channel2].copy()
        Z1 = Z1.reshape((-1, 1))
        Z2 = Z2.reshape((-1, 1))
        Z = np.hstack((Z1, Z2))
        Z = np.float32(Z)
        # obtain centers using kmeans using individual channels
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # apply the segmentation to the image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((A.shape[0], A.shape[1], 2))
        # apply the enhancement if selected
        if Enhance == 0:
            A[:, :, Channel1] = res2[:, :, 0]
        elif Enhance == 1:
            A[:, :, Channel1] = res2[:, :, 1]
        A[:, :, Channel2] = res2[:, :, 1]

    # if the user has selected to cluster by all three channels
    elif len(Channel) == 3:
        Z = A.copy()
        Z = Z.reshape((-1, 3))
        Z = np.float32(Z)
        # obtain centers using kmeans on ALL channels
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # apply the segmentation to the image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((A.shape))
        A[:, :, :] = res2[:, :, :]


'''
The rest of this script is to observe the qualitative segmented image characteristics from k-means segmenting an image based on:

Hue
Saturation
Value
Hue + Saturation
Hue + Value
Saturation + Value

OR

R
G
B
R+G
R+B
G+B
R+G+B

'''
# Read in your image and convert to both HSV and BGR types
img = cv2.imread('flowers.png', 1)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# play around with different parameter values on the HSV image!
segment(img_hsv, (0, 2), 3, 0)
# Convert back to BGR and show image
img1 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
cv2.imshow('HSV segmentation', img1)

# play around with different parameter values on the RGB image!
segment(img, (1, 2), 4, 1)
cv2.imshow('RGB segmentation', img)

# input colors?

cv2.waitKey(0)
