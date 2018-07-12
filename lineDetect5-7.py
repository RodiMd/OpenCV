import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


cap = cv.VideoCapture('/home/rlinux/Desktop/lineDetect/pos/test2.avi')
#cap = cv.VideoCapture('/home/rlinux/Desktop/lineDetect/pos/IGVC2015.mp4')


while cap.isOpened():

	ret, frame = cap.read()
#	hls = cv.cvtColor(frame, cv.COLOR_BGR2HLS)
	hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

	kernel = np.ones((3,3), np.float32)/9

	for i in range (0,3):
		dst = cv.filter2D(hsv, -1, kernel)
		blur = cv.GaussianBlur(dst, (19,19), 0)

#	erosion = cv.erode(blur, kernel, iterations = 1)
	dilation = cv.dilate(blur, kernel, iterations = 2)

#	canny = cv2.Canny(grey,50,100, apertureSize = 5)
	canny = cv.Canny(dilation, 25, 45)

	kernel2 = np.ones((3,3), np.float32)/9
	dilation = cv.dilate(canny, kernel2, iterations = 2)
#	cv.imshow('grey', grey)
#	cv.imshow('thresh', thresh)
#	cv.imshow('hls', hls)
#	cv.imshow('thresh',thresh)
	cv.imshow('hsv', hsv)
	cv.imshow('dilation', dilation)

	k = cv.waitKey(1) & 0XFF
	if k == 27:
		break


cv.destroyAllWindows()
