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
# showimg(hsvMedian, 'hsv median')
        # showimg(hsvMedianCanny, 'hsv median canny')
def averaging(vid):
    blur = cv.blur(vid, (3,3))

    return blur

def gaussian(vid):
    gaussBlur = cv.GaussianBlur(vid, (11,11), 0)

    return gaussBlur

def median(vid):
    medianBlur = cv.medianBlur(vid,3)

    return medianBlur

def bilateralFilter(vid):
    bilateralblur = cv.bilateralFilter(vid, 3, 75, 75)

    return bilateralblur

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

def opening(vid):
    open = cv.morphologyEx(vid, cv.MORPH_OPEN, kernel())

    return open

def closing(vid):
    close = cv.morphologyEx(vid, cv.MORPH_CLOSE, kernel())

    return close

def gradient(vid):
    grad = cv.morphologyEx(vid, cv.MORPH_GRADIENT, kernel())

    return grad

#******************************************************************
#EDGE DETECTION AND IMAGE SHOW

def canny(vid):
    cann = cv.Canny(vid, 40, 80)
    # diCann = cv.dilate(cann, kernel(), iterations = 1)

    return cann

def Laplacian(vid):
    laplacian = cv.Laplacian(vid, cv.CV_64F)

    return laplacian

def Sobel(vid):
    # sobel = cv.Sobel(vid, cv.CV_64F, 0, 1, ksize=7)
    # sobel = cv.Sobel(vid, cv.CV_64F, 1, 0, ksize=7)
    sobel = cv.Sobel(vid, cv.CV_64F, 1, 1, ksize=7)

    return sobel

#*******************************************************************
#MASKING COLOR

def masking(vid):
    lowerLimit = np.array([0, 140, 0])
    upperLimit = np.array([255, 150, 255])

    # lowerLimit = np.array([10, 10, 145])
    # upperLimit = np.array([255, 255, 150])

    mask = cv.inRange(vid, lowerLimit, upperLimit)
    res = cv.bitwise_and(vid, vid, mask=mask)

    return res

#*******************************************************************
#HISTOGRAM EQUALIZER

def histogramEqualizer(vid):

    equalizer = cv.equalizeHist(vid)

    return equalizer

#*******************************************************************
#HOUGH LINE TRANSFORM

def houghLine(vid):

    lines = cv.HoughLinesP(vid, 1, np.pi/180, 50, minLineLength = 20, maxLineGap = 20)
    for line in lines:
        x1, y1, x2, y2 = line[0]

        if -85 < math.degrees(math.atan((y2 - y1) / (x2 - x1))) < 85:
            houghLine = cv.line(vid, (x1, y1), (x2, y2), (0, 255, 0), 0)


    return houghLine

#********************************************************************
#Show image

def showimg(vid, str):
    show = cv.imshow(str, vid)

    return show
