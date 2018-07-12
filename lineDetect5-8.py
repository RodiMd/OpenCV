import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


cap = cv.VideoCapture('/home/rlinux/Desktop/lineDetect/pos/test2.avi')
#cap = cv.VideoCapture('/home/rlinux/Desktop/lineDetect/pos/IGVC2015.mp4')


while cap.isOpened():

	ret, frame = cap.read()
#	hls = cv.cvtColor(frame, cv.COLOR_BGR2HLS)
	hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
	grey = cv.cvtColor(hsv, cv.COLOR_BGR2GRAY)
	hist = cv.equalizeHist(grey)

#	blurHsv = cv.GaussianBlur(hist, (9,9), 0)
	kernel = np.ones((5,5), np.float32)/25

#	erosion = cv.erode(blurHsv, kernel, iterations = 1)
	dst = hist
	for i in range(0,5):
		dst = cv.filter2D(dst, -1, kernel)
		blur = cv.GaussianBlur(dst, (21,21), 0)

#	erosion = cv.erode(blur, kernel, iterations = 1)
	dilation = cv.dilate(blur, kernel, iterations = 2)

	canny = cv.Canny(dilation, 35, 55)

	kernel2 = np.ones((5,5), np.float32)/25
	dilation = cv.dilate(canny, kernel2, iterations = 2)
	cv.imshow('hist', hist)
#	cv.imshow('thresh', thresh)
#	cv.imshow('hls', hls)
#	cv.imshow('thresh',thresh)
	cv.imshow('hsv', hsv)
	cv.imshow('dilation', dilation)
#	cv.imshow('frame', frame)

	k = cv.waitKey(1) & 0XFF
	if k == 27:
		break

cv.destroyAllWindows()
