import cv2 as cv
import time
import PoseModule as pm


cap = cv.VideoCapture('Video/1-P1-720P 高清-AVC.mp4')
# cap = cv.VideoCapture(0)
detector = pm.poseDetector()

pTime = 0
while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        print(lmList[14])
        cv.circle(img, (lmList[14][1], lmList[14][2]), 15, (255, 0, 0), cv.FILLED)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(img, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv.imshow('image', img)
    cv.waitKey(1)