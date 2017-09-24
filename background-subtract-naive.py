import numpy as np
import cv2

filename = 'tiso.avi'
cap = cv2.VideoCapture(filename)
_, bg = cap.read()
bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',cv2.subtract(gray,bg_gray))
    #cv2.imshow('gray',gray)
    #bg_gray = gray;
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
