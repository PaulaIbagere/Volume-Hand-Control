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

detector = htm.handDetector(detectionCon=0.7, maxHands=1)

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

area = 0

colorVol = (255, 0, 0)

while True:
    success, img = cap.read()

    # Find Hand
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=True)
    if lmList:
        # Filter based on size
        area= (bbox[2] - bbox[0])* (bbox[3] - bbox[1])//100
        # print(area)
        
        # When the hand is at the appropriate distance from the screen, find the distance between the index and Thumb
        if 250 < area < 1000:
            
            #Find the distance between the index and Thumb
            length, img, lineInfo = detector.findDistance(4, 8, img)

            # Convert Volume
            volBar = np.interp(length, [50,200], [400, 150])
            volPer = np.interp(length, [50,200], [0, 100])
            
            # Reduce resolution to make it smoother
            # To set the volume incriment
            smoothness = 10
            volPer = smoothness * round(volPer/smoothness)

            # Check fingers up
            fingers = detector.fingersUp()
            print(fingers)

            # if pinky is down set volume
            if not fingers[4]:
                # to set percentage values
                volume.SetMasterVolumeLevelScalar(volPer/100, None)
                cv.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv.FILLED)
                colorVol = (0, 255, 0)
                # time.sleep(0.25)
            else:
                colorVol = (255, 0, 0)

    # Drawings
    cv.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv.FILLED)
    cv.putText(img, f'{int(volPer)}%', (40, 450), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    # Get the current volume
    cVol = volume.GetMasterVolumeLevelScalar() * 100
    cv.putText(img, f'Vol Set: {int(cVol)}', (400, 50), cv.FONT_HERSHEY_COMPLEX, 1, colorVol, 3)

    #Frame rate
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv.putText(img, f'FPS: {int(fps)}', (40, 50), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cv.imshow("Image", img)
    if cv.waitKey(1) & 0xFF == 27:
        break