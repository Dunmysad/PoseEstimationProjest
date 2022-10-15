import cv2 as cv
import numpy as np
import time
import PoseModule as pm

# cap = cv.VideoCapture(0)
cap = cv.VideoCapture('AiTrainer/1-居家哑铃二头只因训练计划-1080P 高清-AVC.mp4')
detector = pm.poseDetector()
count = 0
dir = 0
pTime = 0

while True:

    success, img = cap.read()
    img = cv.resize(img, (1280, 720))

    img = detector.findPose(img,False)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)
    if len(lmList) != 0:
        # Rigth Arm
        angle = detector.findAngle(img, 12, 14, 16)
        # Left Arm
        # angle = detector.findAngle(img, 11, 13, 15)

        per = np.interp(angle, (30, 170), (100, 0))
        bar = np.interp(angle, (30, 170), (100, 650))
        # print(per, angle)

        color = (255, 0, 255)
        if per == 100:
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0:
            color = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0
        print(count)

        # Draw Bar
        cv.rectangle(img, (1100, 100), (1175, 650), color, 3)
        cv.rectangle(img, (1100, int(bar)), (1175, 650), color, cv.FILLED)
        cv.putText(img, f'{int(per)} %', (1100, 75), cv.FONT_HERSHEY_PLAIN, 4, color, 4)

        # Draw Count
        cv.rectangle(img, (0,450), (250,800), (0, 255, 0), cv.FILLED)
        cv.putText(img, str(int(count)), (25, 700), cv.FONT_HERSHEY_PLAIN, 20, (255, 0, 255), 20)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv.putText(img, f'FPS:{int(fps)}', (50, 50), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)

    cv.imshow('Image', img)
    cv.waitKey(1)

