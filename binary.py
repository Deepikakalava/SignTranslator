import cv2
import mediapipe as mp
import time
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

filename = 'finalized_model_n_rf.sav'
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
pTime = 0
cTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    data2 = []
    data3 = []
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                data = np.array([cx])
                data1 = np.array([cy])
                # print((data))
                data2 = np.append(data2, data)
                data3 = np.append(data3, data1)
                #print(id, cx, cy)
                df = pd.DataFrame(data2, columns=['A'])
                df1 = pd.DataFrame(data3, columns=['A'])
                df_transposed = df.transpose()
                df_transposed1 = df1.transpose()
                frames = [df_transposed, df_transposed1]

                result = pd.concat(frames, axis=1)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    if loaded_model.predict(result) == 0:
        cv2.putText(img, str('V'), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)
    else:
        cv2.putText(img, str('not V'), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)
    cv2.rectangle(img, (150, 60), (420, 450), (0, 0, 0), 2)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()