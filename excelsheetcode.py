import cv2
import mediapipe as mp
import time
import numpy as np
import pandas as pd
import xlrd as xlrd
import math
import openpyxl as openpyxl


cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    data2 = []
    data3 = []
    g = np.array([])
    d0 = np.array([])
    d1 = np.array([])
    d2 = np.array([])
    d3 = np.array([])
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                data = np.array([cx])
                data1 = np.array([cy])

                #print((data))
                data2 = np.append(data2, data)
                data3 = np.append(data3, data1)

                #print(data2.ndim)
            df = pd.DataFrame(data2, columns=['A'])
            df1 = pd.DataFrame(data3, columns=['A'])
            df_transposed = df.transpose()
            df_transposed1 = df1.transpose()
            frames = [df_transposed, df_transposed1]

            result = pd.concat(frames, axis=1)
            x0, y0 = handLms.landmark[mpHands.HandLandmark.WRIST].x, handLms.landmark[
                mpHands.HandLandmark.WRIST].y
            x4, y4 = handLms.landmark[mpHands.HandLandmark.THUMB_TIP].x, handLms.landmark[
                mpHands.HandLandmark.THUMB_TIP].y
            x8, y8 = handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x, handLms.landmark[
                mpHands.HandLandmark.INDEX_FINGER_TIP].y
            x12, y12 = handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP].x, \
                       handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP].y
            x16, y16 = handLms.landmark[mpHands.HandLandmark.RING_FINGER_TIP].x, \
                       handLms.landmark[mpHands.HandLandmark.RING_FINGER_TIP].y
            x20, y20 = handLms.landmark[mpHands.HandLandmark.PINKY_TIP].x, handLms.landmark[
                mpHands.HandLandmark.PINKY_TIP].y

            # Calculate the distance between the hand landmarks.
            d0 = math.sqrt((x4 - x0) ** 2 + (y4 - y0) ** 2)
            d1 = math.sqrt((x8 - x0) ** 2 + (y8 - y0) ** 2)
            d2 = math.sqrt((x12 - x0) ** 2 + (y12 - y0) ** 2)
            d3 = math.sqrt((x16 - x0) ** 2 + (y16 - y0) ** 2)
            d4 = math.sqrt((x20 - x0) ** 2 + (y20 - y0) ** 2)
        g = np.append(g, d0)
        g = np.append(g, d1)
        g = np.append(g, d2)
        g = np.append(g, d3)
        g = np.append(g, d4)

        print(g)

        #print(df_transposed)
        #writer = pd.ExcelWriter('e.xlsx', mode="a", engine='openpyxl', if_sheet_exists="overlay")
        #result.to_excel(writer, sheet_name='Sheet1', header=None, startrow=writer.sheets["Sheet1"].max_row, index=False)
        #writer.save()

        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.rectangle(img, (150, 60), (420, 450), (0, 0, 0), 2)


    cv2.imshow("Image", img)
    cv2.waitKey(1)