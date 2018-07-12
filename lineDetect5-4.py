import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


cap = cv.VideoCapture('/home/rlinux/Desktop/lineDetect/pos/for_rodi.avi')

while cap.isOpened():
	
	ret, frame = cap.read()

	hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
#	hls = cv.cvtColor(frame, cv.COLOR_BGR2HLS)
	grey = cv.cvtColor(hsv, cv.COLOR_BGR2GRAY)
#	grey2 = cv.cvtColor(hls, cv.COLOR_BGR2GRAY)

	for i in range (0,2):
		blur = cv.GaussianBlur(hsv, (13,13), 0)
#		blur2 = cv.GaussianBlur(hls, (7,7), 0)

	kernel = np.ones((5,5), np.float32)/25
	dst = cv.filter2D(blur, -1, kernel)
#	dst2 = cv.filter2D(blur2, -1, kernel)

	blur = cv.GaussianBlur(dst, (7,7), 0)
#	blur2 = cv.GaussianBlur(dst2, (5,5), 0)

	erosion = cv.erode(blur, kernel, iterations = 3)
#	erosion2 = cv.erode(blur2, kernel, iterations = 2)

	dilation = cv.dilate(erosion, kernel, iterations =1)
#	dilation2 = cv.dilate(erosion2, kernel, iterations = 1)
#	erosion = cv2.erode(dilation, kernel, iterations = 1)
#	dilation = cv2.dilate(erosion, kernel, iterations = 1)

#	ret, thresh = cv2.threshold(dilation, 127, 255, cv2.THRESH_BINARY_INV)

#	grey = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

#	canny = cv2.Canny(grey,50,100, apertureSize = 5)
	canny = cv.Canny(dilation, 20, 40)
#	canny2 = cv.Canny(dilation2, 20, 40)

	kernel2 = np.ones((5,5), np.float32)/25
	dilation = cv.dilate(canny, kernel2, iterations = 1)
#	dilation2 = cv.dilate(canny2, kernel2, iterations = 1)
#	vid, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#	vid2 = cv2.drawContours(vid, contours, 3, (0,255,0), 3)
#	cv2.imshow('grey', grey)
	cv.imshow('hsv', hsv)
	cv.imshow('dilation', dilation)
	cv.imshow('frame', frame)
#	cv.imshow('hls', hls)
#	cv.imshow('dilation2', dilation2)
	k = cv.waitKey(1) & 0XFF
	if k == 27:
		break



cv.destroyAllWindows()
