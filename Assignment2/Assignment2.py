import cv2
import numpy as np
cap = cv2.VideoCapture(0)


while True:
    rec, frame = cap.read()
    edges = cv2.Canny(frame,100,150)
    cv2.imshow('Frame',edges)
    edge_array = np.column_stack(np.where(edges > 0))
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
cap.release()
cv2.destroyAllWindows()