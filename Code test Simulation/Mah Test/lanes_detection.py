import cv2
from PIL import Image
import numpy as np
import math

def canny_edge_detector(img):
    """Applies the Canny transform"""
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower_yellow = np.array([20, 100, 100], dtype = "uint8")
    upper_yellow = np.array([30, 255, 255], dtype= "uint8")
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray_image, 200, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)

    kernel_size = 5
    blur = cv2.GaussianBlur(mask_yw_image, (kernel_size, kernel_size), 0)  

    low_threshold = 50
    high_threshold = 100
    return cv2.Canny(blur, low_threshold, high_threshold)

def find_vertices(image):
    imshape = image.shape
    lower_left = [0, imshape[0]]
    lower_right = [imshape[1], imshape[0]]
    top_left = [0, imshape[0]/2+imshape[0]/10]
    top_right = [imshape[1],imshape[0]/2+imshape[0]/10]
    vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]
    return vertices

def region_of_interest(image, vertices):
    """We don’t want our car to be paying attention to anything on the horizon, 
    or even in the other lane. 
    Our lane detection pipeline should focus on what’s in front of the car
    Everything out of car view will become black"""
    
    mask = np.zeros_like(image)  
    if len(image.shape) > 2:
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img,lines)
    return line_img

def display_lines(image, lines): 
    line_image = np.zeros_like(image) 
    if lines is not None: 
        for x1, y1, x2, y2 in lines: 
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10) 
    return line_image 

#used below
def get_slope(x1,y1,x2,y2):
    if x1 == x2:
        return 0
    else:
        return (y2-y1)/(x2-x1)

#thick red lines 
def draw_lines(img, lines, color=[255, 0, 0], thickness=6):
    """workflow:
    1) examine each individual line returned by hough & determine if it's in left or right lane by its slope
    because we are working "upside down" with the array, the left lane will have a negative slope and right positive
    2) track extrema
    3) compute averages
    4) solve for b intercept 
    5) use extrema to solve for points
    6) smooth frames and cache
    """
    y_global_min = img.shape[0] #min will be the "highest" y value, or point down the road away from car
    y_max = img.shape[0]
    l_slope, r_slope = [],[]
    l_lane,r_lane = [],[]
    det_slope = 0.4
    α = 0.2 
    #i got this alpha value off of the forums for the weighting between frames.
    #i understand what it does, but i dont understand where it comes from
    #much like some of the parameters in the hough function
    
    for line in lines:
        #1
        for x1,y1,x2,y2 in line:
            slope = get_slope(x1,y1,x2,y2)
            if slope > det_slope:
                r_slope.append(slope)
                r_lane.append(line)
            elif slope < -det_slope:
                l_slope.append(slope)
                l_lane.append(line)
        #2
        y_global_min = min(y1,y2,y_global_min)
    
    # to prevent errors in challenge video from dividing by zero
    if((len(l_lane) == 0) or (len(r_lane) == 0)):

        if len(l_lane) == 0 and len(r_lane) != 0:
            l_slope.append(get_slope(320, 180, 310, 90))
            l_lane.append(np.array([[320, 180, 310, 90]]))
            print(l_lane, r_lane)
            print("l_lane = 0")
        elif len(l_lane) != 0 and len(r_lane) == 0:
            r_slope.append(get_slope(0, 180, 10, 90))
            r_lane.append(np.array([[0, 180, 10, 90]]))
            print(l_lane, r_lane)
            print("r_lane = 0")
        else:
            print ('no lane detected')
            return img,0
        
    #3
    l_slope_mean = np.mean(l_slope,axis =0)
    r_slope_mean = np.mean(r_slope,axis =0)
    l_mean = np.mean(np.array(l_lane),axis=0)
    r_mean = np.mean(np.array(r_lane),axis=0)
    
    if ((r_slope_mean == 0) or (l_slope_mean == 0 )):
        print('dividing by zero')
        return 1
    
    #4, y=mx+b -> b = y -mx
    l_b = l_mean[0][1] - (l_slope_mean * l_mean[0][0])
    r_b = r_mean[0][1] - (r_slope_mean * r_mean[0][0])
    
    #5, using y-extrema (#2), b intercept (#4), and slope (#3) solve for x using y=mx+b
    # x = (y-b)/m
    # these 4 points are our two lines that we will pass to the draw function
    l_x1 = int((y_global_min - l_b)/l_slope_mean) 
    l_x2 = int((y_max - l_b)/l_slope_mean)   
    r_x1 = int((y_global_min - r_b)/r_slope_mean)
    r_x2 = int((y_max - r_b)/r_slope_mean)
    
    #6
    if l_x1 > r_x1:
        l_x1 = int((l_x1+r_x1)/2)
        r_x1 = l_x1
        l_y1 = int((l_slope_mean * l_x1 ) + l_b)
        r_y1 = int((r_slope_mean * r_x1 ) + r_b)
        l_y2 = int((l_slope_mean * l_x2 ) + l_b)
        r_y2 = int((r_slope_mean * r_x2 ) + r_b)
    else:
        l_y1 = y_global_min
        l_y2 = y_max
        r_y1 = y_global_min
        r_y2 = y_max
      
    next_frame = np.array([l_x1,l_y1,l_x2,l_y2,r_x1,r_y1,r_x2,r_y2],dtype ="float32")
    
             
    cv2.line(img, (int(next_frame[0]), int(next_frame[1])), (int(next_frame[2]),int(next_frame[3])), color, thickness)
    cv2.line(img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]),int(next_frame[7])), color, thickness)
    
    top_left = (int(next_frame[0]), int(next_frame[1]))
    bot_left = (int(next_frame[2]), int(next_frame[3]))
    top_right = (int(next_frame[4]), int(next_frame[5]))
    bot_right = (int(next_frame[6]), int(next_frame[7]))

    steering_angle = steering(img, top_left, bot_left, top_right, bot_right)

    return img, steering_angle
    
def process_image(image):
    canny_edges = canny_edge_detector(image)
    
    #find region of interest
    vertices = find_vertices(image)
    roi_image = region_of_interest(canny_edges, vertices)

    lines = cv2.HoughLinesP(roi_image, rho = 2, theta = np.pi/180, threshold = 20,  
                            minLineLength = 20,  
                            maxLineGap = 5) 

    if not isinstance(lines, type(None)):
        line_img = np.zeros_like(image)
        line_img, steering_angle = draw_lines(line_img,lines)
    else:
        line_img = np.zeros_like(image)
        steering_angle = 0

    return line_img, steering_angle

def getAngle(a, b, c):  # mid_point
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    if ang < 0:
        ang += 360
    if ang > 25:
        ang = 25
    return ang

def steering(img, TL, BL, TR, BR): # top_left, bottom_left, top_right, bottom_right
    top_mid = (int((TL[0]+TR[0])/2), int((TL[1] + TR[1])/2))
    bot_mid = (int((BL[0]+BR[0])/2), int((BL[1] + BR[1])/2))     

    cv2.line(img, top_mid, bot_mid, [255,0,0], 3)

    bot_center = (160,180) # frame 320,180
    cv2.line(img, top_mid, bot_center, [0,255,0], 3)   

    A = [top_mid[0], top_mid[1]]
    B = [bot_mid[0], bot_mid[1]]
    C = [bot_center[0], bot_center[1]]

    # print(A, B, C)
    if bot_center > bot_mid:
        angle = -getAngle(C, A, B)  # left angle < 0
    else:
        angle = getAngle(B, A, C)
    print("angle: ", angle)

    return angle

def throttling(angle, speed):
    if abs(angle) == 0:
        return 10
    if abs(angle) < 5:
        return 50
    if abs(angle) < 10:
        return 0
    else:
        if speed > 20:
            return - 10
        else: 
            return 5

def display_line_detection(image):
    steering_angle = 0
    lines, steering_angle = process_image(image)
    combo_image = cv2.addWeighted(image, 0.8, lines, 1, 1)
    cv2.imshow("image", combo_image)
    return steering_angle
