import cv2
import numpy as np
cap = cv2.VideoCapture(0)

#Ofanvarp formula til að reikna frá punkti til línu
def ofanvarp(x1,y1,slope,y_intercept):
    teljari = abs(-slope*x1+y1+y_intercept)
    nefnari = np.sqrt(1+slope**2)
    return teljari/nefnari


def f(delta,N=None,p=None, e = None, s = None):
    #Reikna N fyrir RANSAC ef það er ekki gefið
    if N is None:
        N = np.log(1-p)/np.log(1-(1-e)**s)
        print(N)
    #initialize variable
    while True:
        best_count = 0
        #capture frame 
        _, frame = cap.read()   
        #grab edges with Canny
        edges = cv2.Canny(frame,50,150)
        #remove zero values to grab edge pixel coordinates
        edge_array = np.column_stack(np.where(edges != 0))
        #RANSAC
        #Select N random points
        if edge_array.shape[0] <2:
            print("no edges detexted")
            continue
        elif edge_array.shape[0] < N:
                random_points = edge_array[np.random.choice(edge_array.shape[0],edge_array.shape[0], replace=False)]
                iterations = edge_array.shape[0]
        else:
            random_points = edge_array[np.random.choice(edge_array.shape[0], N, replace=False)]
            iterations = N
        #Iterate throught the points
        for i in range(iterations-1):
            pkt1 = random_points[i,:]
            pkt2 = random_points[i+1,:]
            hallatala = (pkt2[1]-pkt1[1])/(pkt2[0]-pkt1[0])
            y_intercept = pkt1[1]-hallatala*pkt1[0]
            #compute distance to line for each point
            #plug í ofanvarp formula
            x = edge_array[:,0]
            y = edge_array[:,1]
            distances = ofanvarp(x,y,hallatala,y_intercept)
            count_within_delta = np.sum(distances<delta)
            if count_within_delta > best_count:
                best_count = count_within_delta
                bestpoints = np.array([pkt1,pkt2])
        cv2.line(frame, (bestpoints[0,0],bestpoints[0,1]), (bestpoints[1,0],bestpoints[1,1]), (0, 0, 255), 2)
        print(bestpoints)
        cv2.imshow('Frame',frame)
        cv2.imshow('edges',edges)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
f(5,100)
cap.release()
cv2.destroyAllWindows()
