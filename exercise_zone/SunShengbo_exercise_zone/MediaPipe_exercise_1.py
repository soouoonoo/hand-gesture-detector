import cv2
import mediapipe as mp

cap=cv2.VideoCapture(0)

hand_detector=mp.solutions.hands.Hands()

while(1):
    suc,img=cap.read()
    if suc:
        img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        result=hand_detector.process(img_rgb)
        if result.multi_hand_landmarks:
            for handlms in result.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(img,handlms,mp.solutions.hands.HAND_CONNECTIONS)
        cv2.imshow("Camera",img)
    else:
        print("no camera")

    k=cv2.waitKey(1)
    if k==ord('q'):
        break

cap.release()