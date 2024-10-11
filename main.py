
import cv2
import mediapipe as mp
from math import hypot
import pyautogui  # New import for controlling system volume
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands module
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    lmList = []
    if results.multi_hand_landmarks:
        for handlandmark in results.multi_hand_landmarks:
            for id, lm in enumerate(handlandmark.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            mpDraw.draw_landmarks(img, handlandmark, mpHands.HAND_CONNECTIONS)

    if lmList != []:
        # Coordinates of the thumb and index finger
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        # Draw circles and line between thumb and index finger
        cv2.circle(img, (x1, y1), 4, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 4, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Calculate the length between thumb and index finger
        length = hypot(x2 - x1, y2 - y1)
        print(f"Distance between thumb and index: {length}")

        # Adjust volume based on hand distance
        if length > 150:
            print("Volume Up")
            pyautogui.press("volumeup")  # Simulate volume up key press
        elif length < 50:
            print("Volume Down")
            pyautogui.press("volumedown")  # Simulate volume down key press

    # Display the result image
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
