from main import Algorithm
import cv2 as cv
import numpy as np


class Stereorectification(object):
    def __init__(self, left, right, cameraModel:dict = None):
        self.cameraModel = cameraModel
        self.left = left
        self.right = right

        self.left, self.right = self.stereo_rectification()

        # Visualize epilines

    def drawlines(self, img1src, img2src, lines, pts1src, pts2src):
        # img1 - image on which we draw the epilines for the points in img2
        # lines - corresponding epilines
        r, c = img1src.shape
        img1color = cv.cvtColor(img1src, cv.COLOR_GRAY2BGR)
        img2color = cv.cvtColor(img2src, cv.COLOR_GRAY2BGR)
        # Edit: use the same random seed so that two images are comparable!
        np.random.seed(0)
        for r, pt1, pt2 in zip(lines, pts1src, pts2src):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
            img1color = cv.line(img1color, (x0, y0), (x1, y1), color, 1)
            img1color = cv.circle(img1color, tuple(pt1), 5, color, -1)
            img2color = cv.circle(img2color, tuple(pt2), 5, color, -1)
        return img1color, img2color

    def stereo_rectification(self):
        if not self.cameraModel: # 'uncalibrated' using keypoints
            #initiate sift detector
            sift = cv.SIFT_create()

            # find keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(self.left, None)
            kp2, des2 = sift.detectAndCompute(self.right, None)

            # draw keypoints
            imgSift = cv.drawKeypoints(self.left, kp1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # match points on both images
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)  # or pass empty dictionary
            flann = cv.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)

            # keep good matches, calculate distinctive image features
            matchesMask = [[0, 0] for i in range(len(matches))]
            good = []
            pts1 = []
            pts2 = []

            for i, (m, n) in enumerate(matches):
                if m.distance < 0.7 * n.distance:
                    # Keep this keypoint pair
                    matchesMask[i] = [1, 0]
                    good.append(m)
                    pts2.append(kp2[m.trainIdx].pt)
                    pts1.append(kp1[m.queryIdx].pt)

            # Draw the keypoint matches between both pictures
            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 0),
                               matchesMask=matchesMask[300:500],
                               flags=cv.DrawMatchesFlags_DEFAULT)

            keypoint_matches = cv.drawMatchesKnn(self.left, kp1, self.right, kp2, matches[300:500], None, **draw_params)
            #cv.imshow("Keypoint matches", keypoint_matches)

            # Calculate the fundamental matrix for the cameras
            pts1 = np.int32(pts1)
            pts2 = np.int32(pts2)
            fundamental_matrix, inliers = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC)

            # We select only inlier points
            pts1 = pts1[inliers.ravel() == 1]
            pts2 = pts2[inliers.ravel() == 1]

            # # Find epilines corresponding to points in right image (second image) and
            # # drawing its lines on left image
            # lines1 = cv.computeCorrespondEpilines(
            #     pts2.reshape(-1, 1, 2), 2, fundamental_matrix)
            # lines1 = lines1.reshape(-1, 3)
            # img5, img6 = self.drawlines(self.left, self.right, lines1, pts1, pts2)
            #
            # # Find epilines corresponding to points in left image (first image) and
            # # drawing its lines on right image
            # lines2 = cv.computeCorrespondEpilines(
            #     pts1.reshape(-1, 1, 2), 1, fundamental_matrix)
            # lines2 = lines2.reshape(-1, 3)
            # img3, img4 = self.drawlines(self.right, self.left, lines2, pts2, pts1)
            #
            # #cv.imshow('connected', img3)`

            # _, H1, H2 = cv.stereoRectifyUncalibrated(
            #     np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w1, h1)
            # )

            h1, w1 = self.left.shape
            h2, w2 = self.right.shape

            _, H1, H2 = cv.stereoRectifyUncalibrated(
                np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w1, h1)
            )

            # Rectify (undistort) the images and save them
            left_rectified = cv.warpPerspective(self.left, H1, (w1, h1))
            right_rectified = cv.warpPerspective(self.right, H2, (w2, h2))

            return left_rectified, right_rectified

        else: # calibrated using stereocalibration
            # get data from calibrated camera model

            camM1 = self.cameraModel.camera_model['M1'] # camera matrix
            camM2 = self.cameraModel.camera_model['M2'] # camera matrix
            distC1 = self.cameraModel.camera_model['dist1'] # lens distortion coefficient
            distC2 = self.cameraModel.camera_model['dist2'] # lens distortion coefficient
            imgSize = tuple(reversed(self.left.shape)) # reverse tuple cause stereorectify expects w-h not h-w
            R = self.cameraModel.camera_model['R'] # rotation matrix between cameras
            T = self.cameraModel.camera_model['T'] # translation vector
            F = self.cameraModel.camera_model['F'] # fundamental matrix
            r1, r2, p1, p2, q, roi1, roi2 = cv.stereoRectify(cameraMatrix1=camM1, cameraMatrix2=camM2, distCoeffs1=distC1, distCoeffs2=distC2, imageSize=imgSize, R=R, T=T)

            option = cv.CV_16SC2
            map1x, map1y = cv.initUndistortRectifyMap(camM1, distC1, r1, p1, imgSize, option)
            map2x, map2y = cv.initUndistortRectifyMap(camM2, distC2, r2, p1, imgSize, option)

            inter = cv.INTER_LINEAR
            left_rectified = cv.remap(self.left, map1x, map1y, inter)
            right_rectified = cv.remap(self.right, map2x, map2y, inter)


            cv.namedWindow('leftrect', cv.WINDOW_NORMAL)
            cv.imshow('leftrect', left_rectified)
            cv.resizeWindow('leftrect', (600, 600))

            cv.namedWindow('rectright', cv.WINDOW_NORMAL)
            cv.imshow('rectright', right_rectified)
            cv.resizeWindow('rectright', (600, 600))

            return left_rectified, right_rectified

