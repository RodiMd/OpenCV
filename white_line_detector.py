import numpy as np
import cv2 as cv
from Image_Processing import color_ranges, image_processing_task
import math

# Global Variables
ridge_detector = None

def detect_ridges(input_img):
    global ridge_detector
    if ridge_detector is None:
        ridge_detector = cv.ximgproc.RidgeDetectionFilter_create()
    ridges = cv.ximgproc_RidgeDetectionFilter.getRidgeFilteredImage(ridge_detector, _img=input_img)
    return ridges

def get_white_line_detection_task():
    """
    Called Externally to configure the white line detector
    :return: Image_Processing_Task
    """
    which_method_to_use = get_method_name()
    visual = np.array([255, 255, 255], dtype=np.uint8)
    # setup task
    task = image_processing_task.Image_Processing_Task(method=which_method_to_use, obstacle_map_value=1,
                                                       obstacle_visual=visual, clear_negative_obstacle_detections=True)
    return task

def get_method_name():
    """
    passes method name
    Note: no parenthesis after name
    """
<<<<<<< Updated upstream
    return detect_white_lines_ridge

# ----------------------------------------------------------------------------------------------------------------------
=======
    return detect_white_lines3
>>>>>>> Stashed changes


def detect_white_lines_test(img):

    hls = cv.cvtColor(img, cv.COLOR_BGR2HLS)

    mask_white = color_ranges.white_HLS(hls)
    output_white = cv.bitwise_and(img, img, mask=mask_white)
    output = output_white
    imgray = cv.cvtColor(output, cv.COLOR_BGR2GRAY)

    blurred = cv.GaussianBlur(imgray, (5, 5), 0)
    blurred = cv.medianBlur(blurred, 5)
    # blurred = cv.fastNlMeansDenoising(blurred, None, 15, 7, 21)
    # blurred = cv.bilateralFilter(blurred,9,75,75)

    # ret, thresh = cv.threshold(blurred, 100, 255, cv.THRESH_BINARY)
    ret, thresh = cv.threshold(blurred, 100, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    thresh_morph = cv.erode(thresh, kernel, iterations=1)
    thresh_morph = cv.dilate(thresh_morph, kernel, iterations=1)
    thresh_morph = cv.morphologyEx(thresh_morph, cv.MORPH_CLOSE, kernel)
    thresh_morph = cv.dilate(thresh_morph, kernel, iterations=2)
    thresh_morph = cv.morphologyEx(thresh_morph, cv.MORPH_CLOSE, kernel)

    thresh_mask = cv.inRange(thresh_morph, 1, 255)
    return thresh_mask


def detect_white_lines_hybrid(img):
    hls = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    mask_white = color_ranges.white_HLS(hls)
    color_filtered = cv.bitwise_and(img, img, mask=mask_white)
    grey = cv.cvtColor(color_filtered, cv.COLOR_BGR2GRAY)
    ridges = detect_ridges(grey)
    ret, thresh = cv.threshold(ridges, 100, 255, cv.THRESH_BINARY)
    return thresh


def detect_white_lines_ridge(img):
    hls = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    ridges = detect_ridges(hls)
    ret, thresh = cv.threshold(ridges, 100, 255, cv.THRESH_BINARY)
    return thresh


def detect_white_lines2(img):
    #	hls = cv.cvtColor(frame, cv.COLOR_BGR2HLS)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    kernel = np.ones((3, 3), np.float32) / 9

    for i in range(0, 3):
        dst = cv.filter2D(hsv, -1, kernel)
        blur = cv.GaussianBlur(dst, (19, 19), 0)

    #	erosion = cv.erode(blur, kernel, iterations = 1)
    dilation = cv.dilate(blur, kernel, iterations=2)

    #	canny = cv2.Canny(grey,50,100, apertureSize = 5)
    canny = cv.Canny(dilation, 25, 45)

    kernel2 = np.ones((3, 3), np.float32) / 9
    dilation = cv.dilate(canny, kernel2, iterations=2)
    return dilation

def detect_white_lines3(img):

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hls = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # hsv = hls
    hsv = grey

    hsvConv = convolution2D(hsv)
    hsv = hsvConv

    hsvGauss = gaussian(hsv)
    hsv = hsvGauss

    hsvMedian = median(hsv)
    hsv = hsvMedian

    hsvDilate = dilation(hsv)
    hsv = hsvDilate

    hsvErode = erosion(hsv)
    hsv = hsvErode

    hsvThresh = thresh(hsv)
    hsv = hsvThresh

    hsvCanny = canny(hsv)
    hsv = hsvCanny

    hsvDilate = dilation(hsv)
    hsv = hsvDilate

    # hsvMedian = median(hsv)
    # hsv = hsvMedian

    return hsv

#*******************************************************************
#THRESHOLD METHODS

def thresh(vid):
    ret, thresh = cv.threshold(vid, 127, 255, cv.THRESH_BINARY)

    return thresh

#*******************************************************************
#SMOOTHING METHODS

def convolution2D(vid):
    kernel = np.ones((3,3), np.float32)/9
    conv2d = cv.filter2D(vid, -1, kernel)

    return conv2d

def gaussian(vid):
    gaussBlur = cv.GaussianBlur(vid, (11,11), 0)

    return gaussBlur

def median(vid):
    medianBlur = cv.medianBlur(vid,3)

    return medianBlur

#********************************************************************
#MORPHOLOGICAL TRANSFORMATION METHODS

def kernel():
    kernel = np.ones((5,5), np.uint8)
    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7))
    # kernel = cv.getStructuringElement(cv.MORPH_CROSS, (9,9))

    return kernel

def erosion(vid):
    erode = cv.erode(vid, kernel(), iterations = 2)

    return erode

def dilation(vid):

    dilate = cv.dilate(vid, kernel(), iterations = 2)

    return dilate

#******************************************************************
#EDGE DETECTION AND IMAGE SHOW

def canny(vid):
    cann = cv.Canny(vid, 40, 80)
    # diCann = cv.dilate(cann, kernel(), iterations = 1)

    return cann

#********************************************************************
#Show image

def showimg(vid, str):
    show = cv.imshow(str, vid)

    return show
