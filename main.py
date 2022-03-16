import numpy as np
import cv2 as cv
import random, os
from enum import Enum
import Stereocalibration
import Stereorectification
import glob
from PIL import Image
from matplotlib import pyplot as plt
from split import splitImgs_dir


class Algorithm(Enum):
    STEREO_BM = 0
    STEREO_SGBM = 1


def loadImages(isTomato=True, alpha=1.0, beta=0, blur=0):  # does tomatoes on default
    if isTomato:
        path = os.getcwd() + '/splitimgs/'
        imgToSplit = random.choice(os.listdir(path))

        left, right = splitImage(path + imgToSplit)

        cv.namedWindow('original left', cv.WINDOW_KEEPRATIO)
        cv.imshow('original left', left)
        cv.resizeWindow('original left', 800, 600)

        cv.namedWindow('original right', cv.WINDOW_KEEPRATIO)
        cv.imshow('original right', right)
        cv.resizeWindow('original right', 800, 600)


        #left = cv.imread('imgs/left000001.png', 0)
        #right = cv.imread('imgs/right000001.png', 0)

        #rand = random.randrange(0, 5)  # takes random tomato image
        #left = cv.imread('imgs/set' + str(rand) + 'left.png', 0)
        #right = cv.imread('imgs/set' + str(rand) + 'right.png', 0)

        #left = cv.rotate(left, cv.ROTATE_90_COUNTERCLOCKWISE)
        #right = cv.rotate(right, cv.ROTATE_90_COUNTERCLOCKWISE)

    else:
        left = cv.imread('imgs/sleft.jpg', 0)
        right = cv.imread('imgs/sright.jpg', 0)

    #brighten images
    left = cv.equalizeHist(left)
    right = cv.equalizeHist(right)

    if blur > 0:
        # gaussian blur to reduce highlights
        left = cv.GaussianBlur(left, (blur, blur), 0)
        right = cv.GaussianBlur(right, (blur, blur), 0)

    return left, right


def splitImage(path):
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    (h, w) = img.shape[:2]
    centerX = w // 2
    rightimage = img[0:h, centerX:w]
    leftImage = img[0:h, 0:centerX]

    return leftImage, rightimage


def createTrackbars():
    cv.createTrackbar('numDisparities', winName, 1, 17, nothing)
    cv.createTrackbar('blockSize', winName, 5, 50, nothing)
    cv.createTrackbar('uniquenessRatio', winName, 15, 100, nothing)
    cv.createTrackbar('minDisparity', winName, 5, 25, nothing)

    if methodUsed == Algorithm.STEREO_SGBM:
        cv.createTrackbar('P1', winName, 100, 500, nothing)
        cv.createTrackbar('P2', winName, 400, 1100, nothing)


def updateTrackbars():
    # Updating the parameters based on the trackbar positions
    numDisparities = cv.getTrackbarPos('numDisparities', winName) * 16
    blockSize = cv.getTrackbarPos('blockSize', winName) * 2 + 5
    uniquenessRatio = cv.getTrackbarPos('uniquenessRatio', winName)
    minDisparity = cv.getTrackbarPos('minDisparity', winName)

    # Setting the updated parameters before computing disparity map
    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setMinDisparity(minDisparity)

    if methodUsed == Algorithm.STEREO_SGBM:
        stereo.setP1(cv.getTrackbarPos('P1', winName))
        stereo.setP2(cv.getTrackbarPos('P2', winName))

    return numDisparities, minDisparity


def computeDisparity(stereo, recti1, recti2, color = False):
    numdisp, mindisp = updateTrackbars()

    # Calculating disparity using the StereoBM algorithm
    disp = stereo.compute(recti1, recti2)

    # Converting to float32
    disp = disp.astype(np.float32)

    if not color:
        # Scaling down the disparity values and normalizing them
        disp = (disp / 16.0 - mindisp) / numdisp
    else:
        disp = cv.applyColorMap(disp.astype(np.uint8), cv.COLORMAP_JET)

    return disp


def nothing(x):
    pass


if __name__ == '__main__':
    methodUsed = Algorithm.STEREO_SGBM

    imgL_gray, imgR_gray = loadImages(blur=0)

    winName = 'disparity'
    cv.namedWindow(winName, cv.WINDOW_NORMAL)
    cv.resizeWindow(winName, 900, 900)

    createTrackbars()

    # sets default on stereoBM
    stereo = cv.StereoBM_create()

    if methodUsed == Algorithm.STEREO_SGBM:
        stereo = cv.StereoSGBM_create(P1=216,
                                           P2=864,
                                           disp12MaxDiff=1,
                                           speckleWindowSize=100,
                                           speckleRange=32,
                                           preFilterCap=63,
                                           mode=cv.STEREO_SGBM_MODE_SGBM_3WAY)
    else:
        cv.StereoBM()


    # calibrate camera model
    cameraModel = Stereocalibration.StereoCalibration()
    # rectify the images so they're alligned properly
    rectification = Stereorectification.Stereorectification(imgL_gray, imgR_gray, cameraModel)


    while True:
        disparity = computeDisparity(stereo, rectification.left, rectification.right, color=False)
        # Displaying the disparity map
        cv.imshow(winName, disparity)

        # Close window using esc key
        if cv.waitKey(1) == 27:
            break
