import cv2
from PIL import Image
import numpy as np

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

def create_coordinates(image, line_parameters): 
    try:
        slope, intercept = line_parameters
    except TypeError:
        slope, intercept = 0,0

    if ((slope == 0) or (slope == 0 )):
        print('dividing by zero')
        return

    y1 = image.shape[0] 
    y2 = int(y1 * (3 / 5)) 
    x1 = int((y1 - intercept) / slope) 
    x2 = int((y2 - intercept) / slope) 
    return np.array([x1, y1, x2, y2]) 

def display_lines(image, lines): 
    line_image = np.zeros_like(image) 
    if lines is not None: 
        for x1, y1, x2, y2 in lines: 
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10) 
    return line_image 

def average_slope_intercept(image, lines): 
    left_fit = [] 
    right_fit = [] 
    for line in lines: 
        x1, y1, x2, y2 = line.reshape(4) 
          
        # It will fit the polynomial and the intercept and slope 
        parameters = np.polyfit((x1, x2), (y1, y2), 1)  
        slope = parameters[0] 
        intercept = parameters[1]
         
        if slope < 0: 
            left_fit.append((slope, intercept)) 
        else: 
            right_fit.append((slope, intercept)) 
    
    left_line = [0, 0, 0, 0]
    right_line = [0, 0, 0, 0]
    if left_fit:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = create_coordinates(image, left_fit_average)
    if right_fit:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = create_coordinates(image, right_fit_average)

    return np.array([left_line, right_line]) 

def process_image(image):
    canny_edges = canny_edge_detector(image)
    #find region of interest
    vertices = find_vertices(image)
    roi_image = region_of_interest(canny_edges, vertices)

    
    lines = cv2.HoughLinesP(roi_image, rho = 2, theta = np.pi/180, threshold = 40,  
                            minLineLength = 50,  
                            maxLineGap = 50) 
    print("lines", lines)

    if not isinstance(lines, type(None)):
        averaged_lines = average_slope_intercept(image, lines)  
        print("averaged_lines", averaged_lines)
        line_image = display_lines(image, averaged_lines) 
        return line_image
    else:
        return np.zeros_like(image) 

# if __name__ == '__main__':
#     image = Image.open("test2.png")
#     image = np.asarray(image)
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#     process_image(image)
