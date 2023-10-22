import numpy as np
import cv2
import math
import cvzone
from cvzone.ColorModule import ColorFinder  

video_paths = [f'data/Files/Videos/vid ({i}).mp4' for i in range(1, 8)]

### Initialize the Video / usually a webcam number
current_video_index = 0  # Starts from vid (1).mp4
capture = cv2.VideoCapture(video_paths[current_video_index])

### Create colorFinder Object ###
colFinder = ColorFinder(False) #True => debug-mode
hsvVals = {'hmin': 0, 'smin': 144, 'vmin': 0, 'hmax': 74, 'smax': 255, 'vmax': 255}

positionList_x, positionList_y = [], []
x_list = [item for item in range(0,1300)]
prediction = False

number_rects = []

def draw_numbers(img):
    """Draw numbers at the bottom of the image."""
    global number_rects
    number_rects = []  # Clear list before appending new rectangles
    h, w, _ = img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    spacing = 5
    y_pos = h - 30  

    for i in range(7):  
        text = str(i + 1)  
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        x_pos = i * (text_size[0] + spacing)
        cv2.putText(img, text, (x_pos, y_pos), font, font_scale, (255, 255, 255), font_thickness)
        rect = (x_pos, y_pos - text_size[1], text_size[0], text_size[1])
        number_rects.append(rect)

video_index = 4

def setup_video(index):
    global capture
    if capture:
        capture.release()
    capture = cv2.VideoCapture(video_paths[index])

def check_mouse_position_and_switch_video():
    global video_index
    x, y = cv2.getMousePosWindow("imageCOLOR")
    for idx, rect in enumerate(number_rects):
        x1, y1, w, h = rect
        if x1 < x < x1 + w and y1 < y < y1 + h:
            if video_index != idx:
                video_index = idx
                print(f"Switching to video: {video_index}")
                setup_video(video_index)
                break

# Bind mouse callback


#cv2.imshow("imageCOLOR")
#cv2.setMouseCallback("imageCOLOR", mouse_event)

# Setup initial video
setup_video(video_index-1)

while True:
    ### Get the image ###
    success, img = capture.read()
    #img = cv2.imread('data/Files/Ball.png')
    img = img[0:900, :]   #crop image
    check_mouse_position_and_switch_video()  # Add this line inside your while loop

    ### Find the ball based on Color ###
    imgColor, mask = colFinder.update(img,hsvVals)
    ### Location of the ball ###
    imgContours, contours = cvzone.findContours(img, mask, minArea=500)
    draw_numbers(imgContours)

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
            print(X_value)
        if prediction:
            print("SCOOOORE")
            cvzone.putTextRect(imgContours, "SCOOOORE", (50, 150),
                               scale=7,thickness=5,colorR=(0,200,0),offset=20)
        else:
            #print("No basket")
            cvzone.putTextRect(imgContours, "NO BASKET", (50, 150),
                               scale=7,thickness=5,colorR=(0,0,200),offset=20)


    ### Display ###
    imgContours  = cv2.resize(imgContours, (0,0), None, 0.7,0.7 )
    # imgColor  = cv2.resize(mask, (0,0), None, 0.7,0.7 ) #try out with mask
    #cv2.imshow("image", img)
    
    cv2.imshow("imageCOLOR", imgContours)
    cv2.waitKey(100) #frame rate to slow the video

capture.release()
cv2.destroyAllWindows()
