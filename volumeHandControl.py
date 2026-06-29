import cv2 as cv
import time
import numpy as np
import HandTrackingModule as htm
import math
from pycaw.pycaw import AudioUtilities

wCam, hCam = 640, 480

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.7)

device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume
print(f"Audio output: {device.FriendlyName}")
# print(f"- Muted: {bool(volume.GetMute())}")
# print(f"- Volume level: {volume.GetMasterVolumeLevel()} dB")
volumeRange = volume.GetVolumeRange()
# print(f"- Volume range: {volume.GetVolumeRange()[0]} dB - {volume.GetVolumeRange()[1]} dB")

minVol = volumeRange[0]
maxVol = volumeRange[1]

vol = 0
volBar = 400
volPer = 0

while True:
    success, img = cap.read()

    # Find Hand
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if lmList:
        # Filter based on size


        #Find the distance between the index and Thumb

        # Convert Volume
        # Reduce resolution to make it smoother
        # Check fingers up
        # if pinky is down set volume
        # Drawings
        #Frame rate

        #print(lmList[4], lmList[8]) # Print the coordinates of the thumb tip and index finger tip

        x1, y1 = lmList[4][1], lmList[4][2] # Thumb Tip
        x2, y2 = lmList[8][1], lmList[8][2] # Index Finger Tip
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2 # To get the center of the line between the two fingers

        cv.circle(img, (x1, y1), 15, (255, 0, 255), cv.FILLED)
        cv.circle(img, (x2, y2), 15, (255, 0, 255), cv.FILLED)
        cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)

        # To fine the length of the line
        length = math.hypot(x2 - x1, y2 - y1)
        #print(length)

        # Hand range 50 - 300
        # Volume Range -65 - 0

        vol = np.interp(length, [50,300], [minVol, maxVol])
        volBar = np.interp(length, [50,300], [400, 150])
        volPer = np.interp(length, [50,300], [0, 100])
        print(int(length), vol)
        volume.SetMasterVolumeLevel(vol, None)

        # Change color if volume is at minimum
        if length < 50:
            cv.circle(img, (cx, cy), 15, (0, 255, 0), cv.FILLED)

    cv.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv.FILLED)
    cv.putText(img, f'{int(volPer)}%', (40, 450), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv.putText(img, f'FPS: {int(fps)}', (40, 50), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cv.imshow("Image", img)
    if cv.waitKey(1) & 0xFF == 27:
        break