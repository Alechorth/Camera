import cv2 
import numpy as np 

def detect_cherries(img):
    # Captures the live stream frame-by-frame
    no_iteration = True
    cherries_counter = 0
    comparison_list = [0]

    frame = cv2.bitwise_not(img)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([90-10, 200, 150]) 
    upper_red= np.array([90+10, 255, 255])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(hsv, hsv, mask=mask)
    mask = cv2.blur(mask, (3,3))
    
    contours, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 200:
            for zone in comparison_list:
                if area < zone + 300 and area > zone - 300:
                    no_iteration = False
                    cherries_counter += 1
                    comparison_list[comparison_list.index(zone)] = (zone + comparison_list[comparison_list.index(zone)])/2
            if no_iteration:
                comparison_list.append(area)
    
    if cherries_counter > 6:
        print("cherries")
    else:
        print("0")


    cv2.drawContours(image=res, contours=contours, contourIdx=-1, color=(255, 0, 255), thickness=2, lineType=cv2.LINE_AA)
    
    # points are in x,y coordinates
    cv2.imshow('res', res)
    #cv2.imshow('frame', frame)
    #cv2.imshow('mask', mask)
    


# specific color(green here) using OpenCV with Python

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


while (1):
    _, frame = cap.read()
    detect_cherries(frame)

    k = cv2.waitKey(5) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()