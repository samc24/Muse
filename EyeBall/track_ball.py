# https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/

from collections import deque
import numpy as np
import cv2
import imutils
import time
from scipy.optimize import linear_sum_assignment
from KalmanFilter import KalmanFilter, Track, Tracker


# define the lower and upper boundaries of the "green" ball in the HSV color space, then initialize the list of tracked points
# ball_steph2: (10, 174, 138), low = (8, 160, 130), high = (12, 180, 150)
# ball_jordan3 = (7, 154, 86) low = (6,145,75), high = (9, 165, 95) https://imagecolorpicker.com/, https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
ballColorLower = (20,50,110) # (40,130,100) # for 2ktest #(8,150,110) #(3,100,110) # (3,140,75)#(5,140,75)
ballColorUpper = (50,90,160) # (100,200,200) # for 2ktest #(14,255,255)#(10, 170, 95)
ballColorLower_BGR = np.uint8([[list((20, 50, 110))]])  # Lower
ballColorUpper_BGR = np.uint8([[list((90, 255, 255))]])  # Upper

# Convert to HSV
ballColorLower_HSV = cv2.cvtColor(ballColorLower_BGR, cv2.COLOR_BGR2HSV)[0][0]
ballColorUpper_HSV = cv2.cvtColor(ballColorUpper_BGR, cv2.COLOR_BGR2HSV)[0][0]
lower_area = 20 # 350#20
upper_area = 200 # 700#200
dist = 150
frames = 40
trace = 3500

pts = deque(maxlen=64)  # maintain a list of the past N (x, y)-locations of the ball in our video stream. Maintaining such a queue allows us to draw the “contrail” of the ball as its being tracked.

play_name = 'fist21'
video = '/Users/Sameer/Documents/Coding/Muse/EyeBall/videos/spurs_play_'+play_name+'.mp4'
vs = cv2.VideoCapture(video)
hasFrame, frame = vs.read()
output_path = '/Users/Sameer/Documents/Coding/Muse/EyeBall/videos/output'+play_name+'_output.mp4'
vid_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame.shape[1], frame.shape[0]))

# allow the camera or video file to warm up
time.sleep(2.0)

tracker = Tracker(dist, frames, trace, 1)

count=0 #for counting the frames that have passed since the last pass
counter=0 #for counting the pts in the queue
passcount=0 #for counting passes
startCount=30 #for accounting for the first 30 frames
# keep looping until 'q' is pressed or video ends
while True:
    # grab the current frame
    hasFrame, frame = vs.read()
    possiblePass = False
    # reached the end of the video
    if frame is None:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.GaussianBlur(gray, (3, 3), 0) # cv2.blur(gray, (3, 3))

    # resize the frame, blur it, and convert it to the HSV color space
    #frame = imutils.resize(frame, width=1000) # process the frame faster, leading to an increase in FPS
    blurred = cv2.GaussianBlur(frame, (11, 11), 0) # reduce high frequency noise and allow us to focus on the structural objects
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # cv2.imshow("blurred",blurred)
    cv2.imshow("hsv",hsv)
    # construct a mask for the color, then perform a series of dilations and erosions to remove any small blobs left in the mask

    element = None #np.ones((5,5)).astype(np.uint8) # element for morphology, default None
    mask = cv2.inRange(hsv, ballColorLower_HSV, ballColorUpper_HSV) # handles the actual localization of the ball
    mask = cv2.erode(mask, element, iterations=2) # erode and dilate to remove small blobs
    mask = cv2.dilate(mask, element, iterations=2)
    cv2.imshow("mask",mask)

    # find contours in the mask and initialize the current (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    cntr = ([[-1], [-1]])
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid

        # detect near-circles in the hsv filtered image
        contour_list = []
        circularity_list = []
        for contour in cnts:
            epsilon = 0.2*cv2.arcLength(contour,True)
            approx = cv2.approxPolyDP(contour,epsilon,True)
            area = cv2.contourArea(contour)
            # Filter based on length and area
            if (1 < len(approx) < 1000) & (upper_area > area > lower_area):

                contour_list.append(contour)
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour,True)
                circularity = (4*np.pi * area) / (perimeter**2)
                circularity_list.append(circularity)

        cv2.drawContours(frame, contour_list,  -1, (255,0,0), 2)


        # if picking based on circularity
        if contour_list != []:
            most_circ = circularity_list[0]
            c = contour_list[0]
            for i in range(len(circularity_list)):
                if circularity_list[i] > most_circ:
                    most_circ=circularity_list[i]
                    c = contour_list[i]
        else:
            vid_writer.write(frame)
            # show the frame to our screen
            cv2.imshow("Frame", frame)
            cv2.waitKey()
            key = cv2.waitKey(1) & 0xFF

            # if the 'q' key is pressed, stop the loop
            if key == ord("q"):
                break
            continue

        # if picking based on contour area
        #c = max(cnts, key=cv2.contourArea) # largest contour

        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        cntr = ([[cX], [cY]])

        # only proceed if the radius meets a minimum size
        #if radius > 10:
            #if c in contour_list:
            # draw the circle and centroid on the frame, then update the list of tracked points
        cv2.circle(frame, (int(x), int(y)), int(radius),
            (0, 255, 255), 2)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # update the points queue
    pts.appendleft(center)
    centers = []
    counter= counter+1
    if(cntr != ([[-1], [-1]])):
        centers.append(cntr)

    tracker.Update(centers)
    if (len(pts) > 0):
        for i in range(len(tracker.tracks)):
            if (True):
                for j in range(len(tracker.tracks[i].trace)-2):
                    # Draw trace line
                    if(i==0):
                        x1 = tracker.tracks[i].trace[j][0][0]     #get the last 3 points of movement
                        y1 = tracker.tracks[i].trace[j][1][0]
                        x2 = tracker.tracks[i].trace[j+1][0][0]
                        y2 = tracker.tracks[i].trace[j+1][1][0]
                        x3 = tracker.tracks[i].trace[j+2][0][0]
                        y3 = tracker.tracks[i].trace[j+2][1][0]
                        thickness = int(np.sqrt(64 / float(i + 1)) )

                        a = np.array([x1,y1]) #organize as points
                        b = np.array([x2,y2])
                        c = np.array([x3,y3])

                        ba = a - b #for angle calculation
                        bc = c - b
                        thickness = int(np.sqrt(64 / float(i + 1)) )#* 2.5)
                        cv2.line(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0, 255, 0), thickness)
                        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)) #get cosine angle
                        angle = np.arccos(cosine_angle) #get the actual angle in radians
                        angleD = np.degrees(angle) #convert to degrees
                        if((angleD < 60 and angleD > 0) or (angleD < 360 and angleD > 300)): #if the angle b/t the last 3 points is <60 degrees
                            if(count>35 or startCount>0): #if enough frames have passed since last pass was detected (startCount accounts for the beginning 30 frames)
                                print("POSSIBLE PASS MADE")
                                possiblePass = True     #label as possibly a pass
                                count = 0               #reset counter to 0
    #detect direction of balls path (https://www.pyimagesearch.com/2015/09/21/opencv-track-object-movement/)
    if tracker.tracks!=[]:
        for i in np.arange(1, len(tracker.tracks[0].trace)):
            # if either of the tracked points are None, ignore
            # them
            if tracker.tracks[0].trace[i - 1] is None or tracker.tracks[0].trace[i] is None:
                continue
            # check to see if enough points have been accumulated in
            # the buffer
            if counter >= 10 and i == 1 and tracker.tracks[0].trace[-10] is not None:
                # compute the difference between the x and y
                # coordinates and re-initialize the direction
                # text variables
                dX = tracker.tracks[0].trace[-10][0] - tracker.tracks[0].trace[i][0] #get the average x and y movement of ball in the past 10 frames
                dY = tracker.tracks[0].trace[-10][1] - tracker.tracks[0].trace[i][1]
                # ensure there is significant movement in the
                # x-direction
                #if(dY>dX):
                #	possiblePass = False

                if np.abs(dX) > 50:
                    dirX = "East" if np.sign(dX) == 1 else "West"
                    #print(dirX)
                # ensure there is significant movement in the
                # y-direction
                if np.abs(dY) > 120:
                    dirY = "North" if np.sign(dY) == 1 else "South"
                    #print(dirY)
                    if(possiblePass==True):
                        possiblePass = False #if the path of the ball in the last 10 frames is >120 pixels of vertical movement, do not count as a pass
                        #print("too vertical")



    if(possiblePass==True):
        passcount = passcount+1 #update the pass count
    count=count+1 #incrememnt for counting frames between passes
    #print on the pass count on screen
    cv2.putText(frame, "PASSES MADE: " + str(passcount), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    vid_writer.write(frame) #writes to a video
    cv2.imshow("Frame", frame)
    cv2.waitKey()
    key = cv2.waitKey(1) & 0xFF
    startCount = startCount-1
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
    	break

vid_writer.release()
cv2.destroyAllWindows()
