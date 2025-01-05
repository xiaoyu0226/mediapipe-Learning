import cv2
import time
import numpy as np
import handCapture as htm
import os

def main():
    wCam = 1280
    hCam = 960

    # setup video capture and size of capture frame
    cap = cv2.VideoCapture(1)
    cap.set(3, wCam)
    cap.set(4, hCam)

    # Process finger images
    folderPath = "FingerImages"
    myList = os.listdir(folderPath)
    myList = sorted(myList, key=lambda x: int(os.path.splitext(x)[0]))
    overlayList = []

    for imPath in myList:
        # print(f'{folderPath}/{imPath}')
        image = cv2.imread(f'{folderPath}/{imPath}')
        overlayList.append(image)

    # setup hand detector object
    detector = htm.handCapture(maxHands=1, detectionCon=0.8)

    # finger tip index
    tipIds = [4, 8, 12, 16, 20]

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw = False)
        handedness = detector.handedness()    # note we call findPosition and handedness after findHands

        # process the landmarks of the hand
        if len(lmList):
            fingers = []
            # process thumb consider if it is closed
            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1] and handedness == 'Left':
                fingers.append(1)
            elif lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1] and handedness == 'Right':
                fingers.append(1)
            else:
                fingers.append(0)

            # process other 4 fingers consider if it is closed
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            
            number = -1
            if fingers == [1, 1, 1, 1, 1]:
                number = 5
            elif fingers == [0, 1, 1, 1, 1]:
                number = 4
            elif fingers == [0, 1, 1, 1, 0]:
                number = 3
            elif fingers == [0, 1, 1, 0, 0]:
                number = 2
            elif fingers == [0, 1, 0, 0, 0]:
                number = 1
            elif fingers == [0, 0, 0, 0, 0]:
                number = 0
            # overlay the finger image on top of video capture image
            if (number != -1):
                h, w, c = overlayList[number - 1].shape
                img[0: h, 0: w] = overlayList[number - 1]
                
                # draw the number on the screen
                cv2.rectangle(img, (20, 625), (170, 825), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, str(number), (45, 775), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

        cv2.imshow("Finger Counter", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
