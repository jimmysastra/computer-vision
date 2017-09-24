import numpy as np
import cv2

filename = 'omni4.avi'
cap = cv2.VideoCapture(filename)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object
ret, frame = cap.read()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(filename + 'output-bg-subtract.avi',fourcc, 20.0, (1280,720), isColor=False)

#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
#fgbg = cv2.createBackgroundSubtractorMOG2(history = 60, varThreshold=30, detectShadows=False)
fgbg = cv2.createBackgroundSubtractorKNN(history = 500, dist2Threshold = 800.0, detectShadows = False)
ret, frame = cap.read()
fgmask = fgbg.apply(frame, learningRate=0)

while(ret):
    fgmask = fgbg.apply(frame)
    #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    out.write(fgmask)
    #out.write(frame)
    cv2.imshow('frame',fgmask)
    #cv2.imshow('fgmask',frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
