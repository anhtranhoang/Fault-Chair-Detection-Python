import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
from gluoncv import model_zoo, data, utils
import matplotlib.pyplot as plt
import mxnet as mx
import cv2
import time
import numpy as np


def init_weights():  
	ctx = mx.gpu(0)  
	path  = 'F:/HoangAnh/ScanComProject/Training/results/faster_rcnn_fpn_resnet50_v1b_coco_best.params'
	net = model_zoo.get_model('faster_rcnn_fpn_resnet50_v1b_coco', pretrained=False,ctx = [ctx])
	net.classes = ['_background_','back-left leg','back-right leg','front leg','top part','not top part']
	net.load_parameters(path,allow_missing=True, ignore_extra=True,ctx = [ctx])
	print("[Infor] Loaded Parameters ! ")
	return net


def predict(img_side,img_top,net,ctx):
	images = [img_side,img_top]
	c = []
	img1 = cv2.resize(img_top.copy(),(1280,800))
	cv2.imwrite("debug/{}_{}.jpg".format(12,time.time()),img1)
	for i,img in enumerate(images):
		# cv2.imwrite("debug/{}_{}.jpg".format(i,time.time()),img)
		
		img = mx.nd.array(img)
		try:
		   x, orig_img = data.transforms.presets.rcnn.load_test(img)
		except:
			x, orig_img = data.transforms.presets.rcnn.transform_test(img,max_size=1333,short=800)
		
		x = mx.nd.array(x,ctx = ctx)    
		box_ids, scores, bboxes = net(x)
		if i == 1:
			mode = 'top'
			xmin, ymin, xmax, ymax,ax,classes = utils.viz.cv_plot_bbox(orig_img, bboxes[0], scores[0], 
																		box_ids[0], class_names=net.classes,
																		thresh = 0.4,mode = mode)
																		
		else:
			mode = 1
			ax,classes = utils.viz.cv_plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], class_names=net.classes,thresh = 0.8,mode = mode)
		cv2.imwrite("debug/{}_{}.jpg".format(i,time.time()),ax)
		if i == 1:
			if classes == [5] or classes == []:
				stt = False	
			else:
				print(xmin, ymin, xmax, ymax)
				stt = re_check_valid_top(img1[max(ymin-50,0):min(ymax+50,800),max(xmax-303,0):min(xmax+101,1280)])
		c.append(classes)
	# print(c)
	print(stt)
	return c,stt


def re_check_valid_top(img):
	img1 = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	img = cv2.GaussianBlur(img1,(5,5),0)
	img =np.array(img,np.uint8)
	binary_sauvola = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
			cv2.THRESH_BINARY,101,2)
	kernel = np.ones((11,11),np.uint8)
	binary_sauvola = cv2.morphologyEx(binary_sauvola, cv2.MORPH_OPEN, kernel,iterations= 1)
	im2, contours, hierarchy = cv2.findContours(binary_sauvola, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	areas = []
	boo = False
	for cnt in contours:
	# 	areas.append(cv2.contourArea(cnt))
	# reason_area = np.argmin([abs(area -7500) for area in areas])
		epsilon = 0.05*cv2.arcLength(cnt,True)
		approx = cv2.approxPolyDP(cnt,epsilon,True)
		cv2.drawContours(img1, approx, -1, (0,255,0), 3)	
		if abs(cv2.contourArea(cnt))>5000 and abs(cv2.contourArea(cnt)) < 9000:
			if len(approx) == 4:				
				boo =  True
	cv2.imwrite("F:/HoangAnh/ScanComProject/Code/AppDemo/img/{}.jpg".format(time.time()),img1)
	cv2.imwrite("F:/HoangAnh/ScanComProject/Code/AppDemo/img/{}_1.jpg".format(time.time()),binary_sauvola)		
	
	return boo
