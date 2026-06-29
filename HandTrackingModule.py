import cv2 as cv
import mediapipe as mp
import time
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class handDetector():
    def __init__(self, model_path="hand_landmarker.task", maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        # Load model
        base_options = python.BaseOptions(
            model_asset_path=model_path
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=maxHands
        )

        # This creates the actual hand detector object.
        self.detector = vision.HandLandmarker.create_from_options(options)

        self.results = None
        self.tipIds = [4, 8, 12, 16, 20]

        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),       # Index finger
            (5, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
            (9, 13), (13, 14), (14, 15), (15, 16),# Ring finger
            (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
            (0, 17)                               # Palm
        ]


    def findHands(self, frame, draw=True):
        # convert from BGR to RGB because MediaPipe uses RGB format
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # create a Mediapipe image since it can't process opencv images. This converts the OpenCV frame into a MediaPipe image object.
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        self.result = self.detector.detect(mp_image)

        if draw and self.result.hand_landmarks:

            for hand in self.result.hand_landmarks:

                # Get frame and dimensions
                h, w, c = frame.shape

                # store landmark coordinates
                points = []

                for landmark in hand:
                    # convert landmark coordinates to pixels
                    cx = int(landmark.x * w)
                    cy = int(landmark.y * h)

                    points.append((cx, cy))

                    cv.circle(
                        frame,
                        (cx, cy),
                        6,
                        (0, 0, 255),
                        cv.FILLED
                    )
                
                # Draw the connections
                for start_index, end_index in self.HAND_CONNECTIONS:
                    cv.line(frame, points[start_index], points[end_index], (0, 255, 0), 2)
        return frame
    
    def findPosition(self, frame, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []

        if self.result.hand_landmarks:
            hand = self.result.hand_landmarks[handNo]
            h, w, c = frame.shape

            for id, landmark in enumerate(hand):
                cx = int(landmark.x * w)
                cy = int(landmark.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])

                if draw:
                    cv.circle(frame, (cx,cy), 5, (255, 0, 255), cv.FILLED)
            
            # To draw a bounding box around the hand, we need to find the minimum and maximum x and y coordinates of the landmarks. This will give us the top-left and bottom-right corners of the bounding box.
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv.rectangle(frame, (bbox[0]-20, bbox[1]-20), (bbox[2]+20, bbox[3]+20), (0,255,0), 2)
        
        return self.lmList, bbox
    
    def fingersUp(self):
        fingers = []
        
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else: 
            fingers.append(0)

        # 4 fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] -2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
    
    def findDistance(self, p1, p2, frame, draw=True):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2] # Thumb Tip
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2] # Index Finger Tip
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2 # To get the center of the line between the two fingers
        if draw:
            cv.circle(frame, (x1, y1), 15, (255, 0, 255), cv.FILLED)
            cv.circle(frame, (x2, y2), 15, (255, 0, 255), cv.FILLED)
            cv.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv.circle(frame, (cx, cy), 15, (255, 0, 255), cv.FILLED)

        # To fine the length of the line
        length = math.hypot(x2 - x1, y2 - y1)
        return length, frame, [x1, y1, x2, y2, cx, cy]
    
def main():
    pTime = 0
    cTime = 0

    cap = cv.VideoCapture(0)
    detector = handDetector()
    while True:
        # Captures frame and returns success
        success, frame = cap.read()

        if not success:
            break

        frame = detector.findHands(frame)

        lmList = detector.findPosition(frame)
        
        if lmList:
            print("Thumb tip:", lmList[4])
    
        cTime = time.time()
        fps = 1/ (cTime - pTime)
        pTime = cTime

        cv.putText(frame, str(int(fps)), (10, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

        cv.imshow("Hand Tracking", frame)

        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()



if __name__ == "__main__":
    main()