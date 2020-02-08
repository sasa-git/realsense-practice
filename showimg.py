import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)           # カメラFPSを30FPSに設定
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400) # カメラ画像の横幅を1280に設定
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400) # カメラ画像の縦幅を720に設定

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    cv2.imshow('frame2',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()