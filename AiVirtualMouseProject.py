import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy


wCam, hCam = 640, 480
freameR = 100 # Here i define frame reduction as i can move easily below!
smoothing = 7

pTime = 0
plX, plY = 0, 0
clX, clY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
while True:
    # 1 Find Hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)


    # 2. Get the tip of the index and middle fingers
    if len(lmList)!=0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        print(x1, y1, x2, y2)

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        print(fingers)

        cv2.rectangle(img, (freameR, freameR), (wCam - freameR, hCam - freameR), (255, 0, 255), 0)

        # 4. Only index finger : Moving mode
        if fingers[1] == 1 and fingers[2] == 0:

            # 5. Convert our coordinates
            x3 = np.interp(x1, (freameR,wCam-freameR),(0,wScr))
            y3 = np.interp(y1, (freameR, hCam-freameR),(0,hScr))
            # 6. Make Smooth
            clX = plX + (x3-plX) / smoothing
            clY = plY + (y3-plY) / smoothing

            # 7. Move Our Mouse
            autopy.mouse.move(wScr-clX, clY)
            cv2.circle(img,(x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plX, plY = clX, clY

        # 8. Both index and middle fingers are up : Clicking mode
        if fingers[1] == 1 and fingers[2] == 1:

            # 9. Find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12,  img)
            print(length)

            # 10. Click mouse if distance is short
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    # 11. Frame rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,(255,0,0),3)

    # 12. Display

    cv2.imshow("Image", img)
    cv2.waitKey(3)


