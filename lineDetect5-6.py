import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


cap = cv.VideoCapture('/home/rlinux/Desktop/lineDetect/pos/test2.avi')
#cap = cv.VideoCapture('/home/rlinux/Desktop/lineDetect/pos/IGVC2015.mp4')


while cap.isOpened():

	ret, frame = cap.read()
#	hls = cv.cvtColor(frame, cv.COLOR_BGR2HLS)
	hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

	kernel = np.ones((5,5), np.float32)/25
	ret, thresh = cv.threshold(hsv, 127,155,cv.THRESH_TOZERO)

	for i in range (0,3):
		dst = cv.filter2D(thresh, -1, kernel)
		blur = cv.GaussianBlur(dst, (17,17), 0)
#		dst = cv.filter2D(blur, -1, kernel)
#	ret, thresh = cv.threshold(blur, 127, 255, cv.THRESH_TOZERO_INV)

#	kernel = np.ones((9,9), np.float32)/81
#	dst = cv.filter2D(blur, -1, kernel)

#	ret, thresh = cv.threshold(blur, 127, 255, cv.THRESH_TOZERO_INV)

	erosion = cv.erode(blur, kernel, iterations = 2)
	dilation = cv.dilate(erosion, kernel, iterations = 1)

#	canny = cv2.Canny(grey,50,100, apertureSize = 5)
	canny = cv.Canny(dilation, 30, 50)


	kernel2 = np.ones((3,3), np.float32)/9
	dilation = cv.dilate(canny, kernel2, iterations = 1)
#	cv.imshow('grey', grey)
#	cv.imshow('thresh', thresh)
#	cv.imshow('hls', hls)
	cv.imshow('thresh',thresh)
	cv.imshow('hsv', hsv)
	cv.imshow('dilation', dilation)
#	cv.imshow('frame', frame)

	k = cv.waitKey(1) & 0XFF
	if k == 27:
		break



cv.destroyAllWindows()
