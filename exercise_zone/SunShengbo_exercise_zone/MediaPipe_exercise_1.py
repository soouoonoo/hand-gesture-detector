import cv2

cap=cv2.VideoCapture(0)

while(1):
    suc,img=cap.read()
    if suc:
        cv2.imshow("Camera",img)
    else:
        print("no camera")
    k=cv2.waitKey(1)
    if k==ord('q'):
        break