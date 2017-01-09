import math
import sys

import cv2
import numpy as np
from scipy.misc import imsave


class ImageInfo:
    def __init__(self, name, img, position):
        self.name = name
        self.img = img
        self.position = position


def imageBoundingBox(img, M):
    """
       This is a useful helper function that you might choose to implement
       that takes an image, and a transform, and computes the bounding box
       of the transformed image.

       INPUT:
         img: image to get the bounding box of
         M: the transformation to apply to the img
       OUTPUT:
         minX: int for the minimum X value of a corner
         minY: int for the minimum Y value of a corner
         minX: int for the maximum X value of a corner
         minY: int for the maximum Y value of a corner
    """
    #TODO 8
    #TODO-BLOCK-BEGIN
    
    M = M/M[2][2]

    farRight = img.shape[1]-1
    farDown = img.shape[0]-1

    matrixtl = np.zeros((3,1))
    matrixtl[2][0] = 1

    matrixtr = np.zeros((3,1))
    matrixtr[0][0] = farRight
    matrixtr[2][0] = 1

    matrixbl = np.zeros((3,1))
    matrixbl[1][0] = farDown
    matrixbl[2][0] = 1

    matrixbr = np.zeros((3,1))
    matrixbr[0][0] = farRight
    matrixbr[1][0] = farDown
    matrixbr[2][0] = 1

    topLeft = np.dot(M, matrixtl)
    topRight = np.dot(M, matrixtr)
    bottomLeft = np.dot(M, matrixbl)
    bottomRight = np.dot(M, matrixbr)

    xlist = [topLeft[0], topRight[0], bottomLeft[0], bottomRight[0]]
    ylist = [topLeft[1], topRight[1], bottomLeft[1], bottomRight[1]]

    minX = min(xlist)
    maxX = max(xlist)
    minY = min(ylist)
    maxY = max(ylist)
    #TODO-BLOCK-END
    return int(minX), int(minY), int(maxX), int(maxY)

def accumulateBlend(img, acc, M, blendWidth):
    """
       INPUT:
         img: image to add to the accumulator
         acc: portion of the accumulated image where img should be added
         M: the transformation mapping the input image to the accumulator
         blendWidth: width of blending function. horizontal hat function
       OUTPUT:
         modify acc with weighted copy of img added where the first
         three channels of acc record the weighted sum of the pixel colors
         and the fourth channel of acc records a sum of the weights
    """
    # BEGIN TODO 10
    # Fill in this routine
    #TODO-BLOCK-BEGIN
    def interpolate(channel, accImg, img):
        '''Bilinear Interpolation'''
        #get floor and ceiling of accImgList first two coor
        xFloor = int(np.floor(accImg[0]))
        xCeil = int(np.ceil(accImg[0]))
        yFloor = int(np.floor(accImg[1]))
        yCeil = int(np.ceil(accImg[1]))

        #interpolate between top x (find value)
        if xCeil == xFloor:
            fx1 = img[yFloor][xFloor][channel]
        else:
            xInter = (accImg[0]-xFloor)/(xCeil-xFloor)
            fx1 = (1-xInter)*img[yFloor][xFloor][channel]    +    xInter*img[yFloor][xCeil][channel]
        #interpolate between bottom x(find value)
        if xCeil == xFloor:
            fx2 = img[yCeil][xFloor][channel]
        else:
            fx2 = (1-xInter)*img[yCeil][xFloor][channel]    +    xInter*img[yCeil][xCeil][channel]
        #interpolate between first 2 interpolations (find value)
        if yCeil == yFloor:
            imgRGB_OnePixel = fx1
        else:
            yInter = (accImg[1]-yFloor)/(yCeil-yFloor)
            imgRGB_OnePixel = (1-yInter)*fx1     +     yInter*fx2

        return imgRGB_OnePixel


    def blending(channel, accImg, img, k, blendWidth, acc, imgRGB_OnePixel, row, col,j):
        '''Blending'''
        #if first coord (x) is within boundary 
        #then blend acc values with img values
        farX = (img.shape[1]-1)-blendWidth
            
        #Left blend
        if (accImg[0] < blendWidth):
            k1 = 1-(blendWidth-accImg[0])/blendWidth
            acc[row][col][channel] = ((1-k1)*acc[row][col][channel] + (k1)*imgRGB_OnePixel)
            k1 = k1 + (1-k1)*acc[row][col][3]
        #Right blend
        elif (accImg[0] > farX):
            k1 = ((img.shape[1]-1)-accImg[0])  /  blendWidth
            acc[row][col][channel] = k1*imgRGB_OnePixel  +   (1-k1)*acc[row][col][channel]
            k1 = k1 + (1-k1)*acc[row][col][3]
        #No blend
        else:
            k1 = 1
            acc[row][col][channel] = imgRGB_OnePixel

        k = k1
        return acc, k   

    minX, minY, maxX, maxY = imageBoundingBox(img, M)

    M_inv = np.linalg.inv(M)
    M_inv /= M_inv[2][2]

    j = 0

    for row in range(minY,maxY):
        # if row%50 != 1:
        #     continue
        for col in range(minX, maxX):
            # if col%50 != 5:
            #     continue
            j += 1

            acc_coor= np.array([[col],[row],[1]])
            accImg = np.dot(M_inv, acc_coor)
            accImg /= accImg[2][0]
            img.astype(dtype = np.uint8)

            k = 0

            xFloor = int(np.floor(accImg[0]))
            xCeil = int(np.ceil(accImg[0]))
            yFloor = int(np.floor(accImg[1]))
            yCeil = int(np.ceil(accImg[1]))

            '''Check if in image bounds'''
            if (xCeil<img.shape[1]) and (yCeil<img.shape[0]) and (xFloor>-1) and (yFloor>-1):

                '''Checking black pixels'''
                c1 = img[yFloor][xFloor]
                c2 = img[yCeil][xFloor]
                c3 = img[yFloor][xCeil]
                c4 = img[yCeil][xCeil]
                clist = []
                clist.extend([c1, c2, c3, c4])
                noblacks = True
                for c in clist:
                    if (np.count_nonzero(c) == 0):
                        noblacks = False
                        break

                '''If no blacks: do this'''
                if (noblacks == True):
                    for channel in range(3):
                        imgRGB_OnePixel = interpolate(channel, accImg, img)

                        acc, k = blending(channel, accImg, img, k, blendWidth, acc, imgRGB_OnePixel, row, col,j)

                    acc[row][col][3] = k
    #TODO-BLOCK-END
    # END TODO

def normalizeBlend(acc):
    """
       INPUT:
         acc: input image whose alpha channel (4th channel) contains
         normalizing weight values
       OUTPUT:
         img: image with r,g,b values of acc normalized
    """
    # BEGIN TODO 11
    # fill in this routine..
    #TODO-BLOCK-BEGIN
    img = acc[:,:,:3]/np.maximum(acc[:,:,3:], 1e-7)
    #TODO-BLOCK-END
    # END TODO
    return img


def blendImages(ipv, blendWidth, is360=False, A_out=None):
    """
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         blendWidth: width of the blending function
       OUTPUT:
         croppedImage: final mosaic created by blending all images and
         correcting for any vertical drift
    """
    # Compute bounding box for the mosaic
    minX = sys.maxint
    minY = sys.maxint
    maxX = 0
    maxY = 0
    channels = -1
    width = -1  # Assumes all images are the same width
    M = np.identity(3)
    for i in ipv:
        M = i.position
        img = i.img
        _, w, c = img.shape
        if channels == -1:
            channels = c
            width = w

        # BEGIN TODO 9
        # add some code here to update minX, ..., maxY
        #TODO-BLOCK-BEGIN
        rminX, rminY, rmaxX, rmaxY = imageBoundingBox(img, M)

        if rminX < minX:
            minX = rminX
        if rminY < minY:
            minY = rminY
        if rmaxX > maxX:
            maxX = rmaxX
        if rmaxY > maxY:
            maxY = rmaxY
        #TODO-BLOCK-END
        # END TODO

    # Create an accumulator image
    accWidth = int(math.ceil(maxX) - math.floor(minX))
    accHeight = int(math.ceil(maxY) - math.floor(minY))
    print 'accWidth, accHeight:', (accWidth, accHeight)
    acc = np.zeros((accHeight, accWidth, channels + 1),dtype = np.float16)
    # Add in all the images
    translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])
    M = np.identity(3)
    for count, i in enumerate(ipv):
        M = i.position
        img = i.img

        M_trans = translation.dot(M)
        accumulateBlend(img, acc, M_trans, blendWidth)

        # First image
        if count == 0:
            p = np.array([0.5 * width, 0, 1])
            p = M_trans.dot(p)
            x_init, y_init = p[:2] / p[2]
        # Last image
        if count == (len(ipv) - 1):
            p = np.array([0.5 * width, 0, 1])
            p = M_trans.dot(p)
            x_final, y_final = p[:2] / p[2]

    compImage = normalizeBlend(acc)

    # Determine the final image width
    outputWidth = (accWidth - width) if is360 else accWidth
    # Compute the affine transform
    A = np.identity(3)
    # BEGIN TODO 12
    # fill in appropriate entries in A to trim the left edge and
    # to take out the vertical drift if this is a 360 panorama
    # (i.e. is360 is true)
    # Shift it left by the correct amount
    # Then handle the vertical drift
    # Note: warpPerspective does inverse mapping which means A is an affine
    # transform that maps final panorama coordinates to accumulator coordinates
    #TODO-BLOCK-BEGIN
    if is360:
        #align y coord, im1 and imfinal y coor equal
        A[0][2] =  -x_final      #x_final-x_init
        A[1][0] = -np.float(y_final-y_init)/np.float(x_final-x_init)
    A.astype(dtype = np.uint8)
    compImage = compImage.astype(dtype = np.uint8)
    #TODO-BLOCK-END
    # END TODO

    if A_out is not None:
        A_out[:] = A

    # Warp and crop the composite

    croppedImage = cv2.warpPerspective(
        compImage, A, (outputWidth, accHeight), flags=cv2.INTER_LINEAR
    )

    return croppedImage

