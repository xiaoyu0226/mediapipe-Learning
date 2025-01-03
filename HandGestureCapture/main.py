import cv2
import time
import numpy as np
import handCapture as htm
import math
import subprocess

def main():
    wCam = 640
    hCam = 480
    pTime = 0
    vol = 0
    volBar = 400
    volPer = 0
    
    # setup video capture and size of capture frame
    cap = cv2.VideoCapture(1)
    cap.set(3, wCam)
    cap.set(4, hCam)

    # setup hand detector object
    detector = htm.handCapture(maxHands=1, detectionCon=0.8)

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw = False)

        # process the landmarks of the hand
        if len(lmList):
            # landmark index 4 is tip of thumb
            # landmark index 8 is tip of index finger 
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]

            # mark the center of the distance between two finger tips
            cx, cy = (x1 + x2) //2, (y1 + y2) //2

            # draw the finger tips landmarks and center circle
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

            length = math.hypot(x2 - x1, y2 - y1)    # get the length 

            # get the height of volume bar 
            volBar = np.interp(length, [50, 250], [400, 150])
            # get the volume percentage 
            volPer = np.interp(length, [50, 250], [0, 100])
            
            # change computer volume (this is MAC only)
            script = f"osascript -e 'set volume output volume {int(volPer)}'"
            subprocess.run(script, shell=True)

            # if the distance between finger tips too small, make a snap visual effect 
            if length < 50:
                cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

        # draw the volume bar 
        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)

        # write out the volume percentage 
        cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

        # write the fps (can comment out not necessary)
        # cTime = time.time()
        # fps = 1 / (cTime - pTime)
        # pTime = cTime
        # cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow("Volume Controller", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
