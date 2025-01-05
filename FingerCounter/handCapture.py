import cv2
import mediapipe as mp
import time

class handCapture():
    def __init__(self, mode = False, maxHands = 2, complexity = 1, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode 
        self.maxHands = maxHands
        self.complexity = 1
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)    # hands object can only process RGB image
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handlms, self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self, img, handNo = 0, draw = True):
        lmList = []
        if self.results.multi_hand_landmarks:   # use this to confirm hand detected
            myHand = self.results.multi_hand_landmarks[handNo]    # myHand consists of 21 landmark information
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return lmList
    
    def handedness(self, handNo = 0):
        handedness = None
        if self.results.multi_handedness:
            myHand = self.results.multi_handedness[handNo]
            handedness = myHand.classification[0].label

            # need to consider if palm or back of hand 
            palm = self.results.multi_hand_landmarks[handNo].landmark[0]
            wrist = self.results.multi_hand_landmarks[handNo].landmark[1]
            if (handedness == 'Left' and palm.z > wrist.z) or (handedness == 'Right' and palm.z < wrist.z):
                return 'Left'
            else:
                return 'Right'

        return handedness

    

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(1)
    detector = handCapture()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime 
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
