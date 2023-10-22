import numpy as np
import cv2
import math
import cvzone
from cvzone.ColorModule import ColorFinder  

### Initialize the Video / usually a webcam number
capture = cv2.VideoCapture('data/Files/Videos/vid (4).mp4')
# data/Files/Ball.png

### Create colorFinder Object ###
colFinder = ColorFinder(False) #True => debug-mode

# Varivables 
hsvVals = {'hmin': 0, 'smin': 144, 'vmin': 0, 'hmax': 74, 'smax': 255, 'vmax': 255}
positionList_x, positionList_y = [], []
x_list = [item for item in range(0,1300)]
prediction = False



while True:
    ### Get the image ###
    success, img = capture.read()
    #img = cv2.imread('data/Files/Ball.png')
    img = img[0:900, :]   #crop image


    ### Find the ball based on Color ###
    imgColor, mask = colFinder.update(img,hsvVals)
    ### Location of the ball ###
    imgContours, contours = cvzone.findContours(img, mask, minArea=500)

    if contours: 
        #cx, cy = contours[0]['center'] # in contours sorted=True
        # cv2.circle(imgContours, (cx, cy), 5, (0,255,0), None,0.7,0.7)
        # positionList.append(contours[0]['center'])

        # separate coordinates for prediction
        positionList_x.append(contours[0]['center'][0])
        positionList_y.append(contours[0]['center'][1])


    ### polynomial fit line ### 
    if positionList_x:
        a,b,c = np.polyfit(positionList_x,positionList_y, 2) # coefficient f(x) = ax^2 + bx + c
        for (x,y) in zip(positionList_x,positionList_y):
            position = (x,y)
            cv2.circle(imgContours, position, 6, (0,0,255), cv2.FILLED)


        for x in x_list:
            y = int(a * x ** 2 + b*x + c)
            cv2.circle(imgContours, (x,y), 2, (0,255,0), cv2.FILLED)

        if len(positionList_x) < 10:
            ### Prediction of the basket
            # x & y values of basket:
            #   | 330-430 | 
            #   ##########  Y:–––590–––
            #    ########
            #     #####
            A = a
            B = b
            C = c-590
            X_value = int((-B - math.sqrt(B**2-(4*A*C)))/(2*A))
            prediction = 330 < X_value < 430
            # print(X_value)
        if prediction:
            print("SCOOOORE")
            cvzone.putTextRect(imgContours, "SCOOOORE", (50, 150),
                               scale=7,thickness=5,colorR=(0,200,0),offset=20)
        else:
            print("No basket")
            cvzone.putTextRect(imgContours, "NO BASKET", (50, 150),
                               scale=7,thickness=5,colorR=(0,0,200),offset=20)


    ### Display ###
    imgContours  = cv2.resize(imgContours, (0,0), None, 0.7,0.7 )
    # imgColor  = cv2.resize(mask, (0,0), None, 0.7,0.7 ) #try out with mask
    #cv2.imshow("image", img)
    
    cv2.imshow("imageCOLOR", imgContours)
    cv2.waitKey(100) #frame rate to slow the video


