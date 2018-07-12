import cv2
import numpy as np
import matplotlib.pyplot as plt


cap = cv2.VideoCapture('/home/rlinux/Desktop/lineDetect/pos/IGVC2015.mp4')

while cap.isOpened():
	
	ret, frame = cap.read()

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	grey = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
	for i in range (0,4):
		blur = cv2.GaussianBlur(hsv, (17,17), 0)

	kernel = np.ones((11,11), np.float32)/121
	dst = cv2.filter2D(blur, -1, kernel)
	blur = cv2.GaussianBlur(dst, (17,17), 0)
	erosion = cv2.erode(blur, kernel, iterations = 2)
	dilation = cv2.dilate(erosion, kernel, iterations = 2)
#	erosion = cv2.erode(dilation, kernel, iterations = 1)
#	dilation = cv2.dilate(erosion, kernel, iterations = 1)

#	ret, thresh = cv2.threshold(dilation, 127, 255, cv2.THRESH_BINARY_INV)

#	grey = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

#	canny = cv2.Canny(grey,50,100, apertureSize = 5)
	canny = cv2.Canny(dilation, 20, 40)
	kernel2 = np.ones((7,7), np.float32)/49
	dilation = cv2.dilate(blur, kernel2, iterations = 3)
#	vid, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#	vid2 = cv2.drawContours(vid, contours, 3, (0,255,0), 3)
#	cv2.imshow('grey', grey)
	cv2.imshow('canny', canny)
	cv2.imshow('dilation', dilation)
	cv2.imshow('frame', frame)
	k = cv2.waitKey(5) & 0XFF
	if k == 27:
		break



cv2.destroyAllWindows()
