import numpy as np
import cv2 as cv
from PIL import Image
from split import splitImg_var
import glob
import argparse
import os
import csv


class StereoCalibration:
    def __init__(self):
        # termination criteria
        self.criteria = (cv.TERM_CRITERIA_EPS +
                         cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv.TERM_CRITERIA_EPS +
                             cv.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((9 * 6, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.

        self.read_images()

    def read_images(self):

        images_right = glob.glob(os.getcwd() + '/calibration/left/*.png')
        images_left = glob.glob(os.getcwd() + '/calibration/right/*.png')

        for i, fname in enumerate(images_right):
            img_l = cv.imread(images_left[i])
            img_r = cv.imread(images_right[i])

            gray_l = cv.cvtColor(img_l, cv.COLOR_BGR2GRAY)
            gray_r = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)

            print('chessboard ' + str(i))
            # Find the chess board corners
            ret_l, corners_l = cv.findChessboardCorners(gray_l, (7, 9), None)
            print(str(i) + ' done out of ' + str(len(images_right)))
            ret_r, corners_r = cv.findChessboardCorners(gray_r, (7, 9), None)
            print('done chessboards ' + str(i))
            # If found, add object points, image points (after refining them)
            self.objpoints.append(self.objp)

            # if ret_l is True:
            #     rt = cv.cornerSubPix(gray_l, corners_l, (11, 11),
            #                           (-1, -1), self.criteria)
            #     self.imgpoints_l.append(corners_l)
                # # Draw and display the corners
                # ret_l = cv.drawChessboardCorners(img_l, (9, 6),
                #                                   corners_l, ret_l)
                # cv.imshow(images_left[i], img_l)
                # cv.waitKey()

            # if ret_r is True:
            #     rt = cv.cornerSubPix(gray_r, corners_r, (11, 11),
            #                           (-1, -1), self.criteria)
            #     self.imgpoints_r.append(corners_r)
                # Draw and display the corners
                # ret_r = cv.drawChessboardCorners(img_r, (9, 6),
                #                                   corners_r, ret_r)
                # cv.imshow(images_right[i], img_r)
                # cv.waitKey()

            img_shape = gray_l.shape[::-1]

        rt, self.M1, self.d1, self.r1, self.t1 = cv.calibrateCamera(
            self.objpoints, self.imgpoints_l, img_shape, None, None)
        rt, self.M2, self.d2, self.r2, self.t2 = cv.calibrateCamera(
            self.objpoints, self.imgpoints_r, img_shape, None, None)

        self.camera_model = self.stereo_calibrate(img_shape)

    def stereo_calibrate(self, dims):
        flags = 0
        flags |= cv.CALIB_FIX_INTRINSIC
        #flags |= cv.CALIB_FIX_PRINCIPAL_POINT
        # flags |= cv.CALIB_USE_INTRINSIC_GUESS
        flags |= cv.CALIB_FIX_FOCAL_LENGTH
        flags |= cv.CALIB_FIX_ASPECT_RATIO
        flags |= cv.CALIB_ZERO_TANGENT_DIST
        # flags |= cv.CALIB_RATIONAL_MODEL
        flags |= cv.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv.CALIB_FIX_K3
        # flags |= cv.CALIB_FIX_K4
        # flags |= cv.CALIB_FIX_K5

        stereocalib_criteria = (cv.TERM_CRITERIA_MAX_ITER +
                                cv.TERM_CRITERIA_EPS, 100, 1e-5)

        # self.camM1 = np.array([[1400, -0.165, 960],
        #                   [0.0, 1400, 540],
        #                   [0.0, 0.0, 0.0]])
        #
        # self.camM2 = np.array([[1400, 0.0, 960],
        #                        [0.0, 1400, 540],
        #                        [0.0, 0.0, 0.0]])

        ret, M1, d1, M2, d2, R, T, E, F = cv.stereoCalibrate(
            self.objpoints, self.imgpoints_l,
            self.imgpoints_r, self.M1, self.d1, self.M2,
            self.d2, dims,
            criteria=stereocalib_criteria, flags=flags)

        camera_model = dict([('M1', M1), ('M2', M2), ('dist1', d1),
                             ('dist2', d2), ('rvecs1', self.r1),
                             ('rvecs2', self.r2), ('R', R), ('T', T),
                             ('E', E), ('F', F)])

        w = csv.writer(open('cameraModel.csv', 'w'))
        for key, val in camera_model.items():
            w.writerow([key, val])

        return camera_model
