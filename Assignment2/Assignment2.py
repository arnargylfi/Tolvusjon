import cv2
import numpy as np
cap = cv2.VideoCapture(0)
import time


def f(delta,N=None,p=None, e = None, s = None):
    fps = 0
    #Reikna N fyrir RANSAC ef það er ekki gefið
    if N is None:
        N = np.log(1-p)/np.log(1-(1-e)**s)
        print(N)
    #initialize variable
    while True:
        tic = time.time()
        best_count = 0
        #capture frame 
        _, frame = cap.read()   
        #grab edges with Canny
        edges = cv2.Canny(frame,100,200)
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

            a = pkt2[1] - pkt1[1]
            b = pkt1[0] - pkt2[0]
            c = pkt2[0] * pkt1[1] - pkt1[0] * pkt2[1]

            # Normalize the line parameters
            normalize = np.sqrt(a**2 + b**2)
            a = a/normalize
            b = b/normalize 
            c = c/normalize

            # Compute distances of all points to the line
            distances = np.abs(a * edge_array[:, 0] + b * edge_array[:, 1] + c)
            
            #compute distance to line for each point
            #plug í ofanvarp formula
            count_within_delta = np.sum(distances<delta)
            if count_within_delta > best_count:
                best_count = count_within_delta
                bestpoints = np.array([pkt1,pkt2])
        cv2.line(frame, (bestpoints[0,1],bestpoints[0,0]), (bestpoints[1,1],bestpoints[1,0]), (0, 0, 255), 2)
        for point in random_points:
            cv2.circle(frame,(point[1],point[0]),2,(0,0,255),-1)
        cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Frame',frame)
        cv2.imshow('edges',edges)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        toc = time.time()
        fps = 1/(toc-tic)

f(2,50)
cap.release()
cv2.destroyAllWindows()
