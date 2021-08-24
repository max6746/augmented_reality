import cv2
import numpy as np
import pyglet

vid_path = 'video.mp4'
player = pyglet.media.Player()
MediaLoad = pyglet.media.load(vid_path)
player.queue(MediaLoad)
player.loop = True


cap = cv2.VideoCapture(0)
imgTar = cv2.imread('trgimg.jpg')
# imgTarGry = cv2.cvtColor(imgTar, cv2.COLOR_BGR2GRAY)
# cv2.imshow('img_gray', imgTarGry)
# cv2.waitKey(0)


myVid = cv2.VideoCapture('video.mp4')


success, imgVideo = myVid.read()
hT,wT,cT = imgTar.shape
imgVideo = cv2.resize(imgVideo,(wT,hT))

detection = False
frameCounter = 0

orb = cv2.ORB_create(nfeatures=1000)
# kp1, des1 = orb.detectAndCompute(imgTarGry, None)
# imgTarGry = cv2.drawKeypoints(imgTarGry,kp1,None)

kp1, des1 = orb.detectAndCompute(imgTar, None)
imgTar = cv2.drawKeypoints(imgTar,kp1,None)

while True:

    success, imgWebcam = cap.read()
    imgAug = imgWebcam.copy()
    # imgWebcamGry = cv2.cvtColor(imgWebcam, cv2.COLOR_BGR2GRAY)
    # kp2, des2 = orb.detectAndCompute(imgWebcamGry, None)
    # imgWebcamGry = cv2.drawKeypoints(imgWebcamGry,kp2,None)
    kp2, des2 = orb.detectAndCompute(imgWebcam, None)
    imgWebcam = cv2.drawKeypoints(imgWebcam,kp2,None)

    if detection == False:
        myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frameCounter = 0
        player.queue(MediaLoad)
        player.play()
    else:
        if frameCounter == myVid.get(cv2.CAP_PROP_FRAME_COUNT):
            myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0
            player.queue(MediaLoad)
            player.play()
        success, imgVideo = myVid.read()
        imgVideo = cv2.resize(imgVideo,(wT,hT))


    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good =[]
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    print(len(good))
    # imgFeatures = cv2.drawMatches(imgTarGry, kp1, imgWebcamGry, kp2, good, None, flags=2)
    imgFeatures = cv2.drawMatches(imgTar, kp1, imgWebcam, kp2, good, None, flags=2)

    if len(good) > 5:
        detection = True
        srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        # matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 2)
        matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.LMEDS)
        print(matrix)

        pts = np.float32([[0,0], [0,hT], [wT,hT], [wT,0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, matrix)
        img2 = cv2.polylines(imgWebcam, [np.int32(dst)], True, (255,0,255), 3)

        imgWarp = cv2.warpPerspective(imgVideo, matrix, (imgWebcam.shape[1], imgWebcam.shape[0]))

        maskNew = np.zeros((imgWebcam.shape[0], imgWebcam.shape[1]), np.uint8)
        cv2.fillPoly(maskNew, [np.int32(dst)], (255,255,255))
        maskInv = cv2.bitwise_not(maskNew)
        imgAug = cv2.bitwise_and(imgAug, imgAug, mask=maskInv)
        imgAug = cv2.bitwise_or(imgWarp, imgAug)

        # imgStacked = stackImages(([imgWebcam, imgWarp], [imgWebcam, imgWarp]), 0.5)

    cv2.imshow('imgAug', imgAug)
    # cv2.imshow('imgWarp', imgWarp)
    # cv2.imshow('img2', img2)
    # cv2.imshow('imgFeatures', imgFeatures)

    # cv2.imshow('ImgTarget', imgTar)
    # cv2.imshow('myVid', imgVideo)
    # cv2.imshow('webcam', imgWebcam)
    cv2.waitKey(1)
    frameCounter += 1
