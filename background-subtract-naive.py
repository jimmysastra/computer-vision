import numpy as np
import cv2

filename = 'omni4.avi'
cap = cv2.VideoCapture(filename)
_, bg = cap.read()
bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
ret, frame = cap.read()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(filename + 'output-bg-subtract-naive.avi',fourcc, 20.0, (1280,720), isColor=False)

#while(cap.isOpened()):
while(ret):
    # ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.subtract(gray,bg_gray)
    out.write(frame)
    cv2.imshow('frame', frame)
    #cv2.imshow('gray',gray)
    #bg_gray = gray;
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
