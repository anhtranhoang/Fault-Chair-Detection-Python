import json
import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'
os.environ['MXNET_GPU_MEM_POOL_ROUND_LINEAR_CUTOFF'] = '26'
os.environ['MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD'] = '999'
os.environ['MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD'] = '25'
os.environ['MXNET_GPU_COPY_NTHREADS'] = '4'
os.environ['MXNET_OPTIMIZER_AGGREGATION_SIZE'] = '54'
import UI.bg
import sys
import time
from shutil import copy

import cv2
import mxnet as mx
import numpy as np
from pypylon import pylon
import PyQt5
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtGui import QImage, QPixmap
from prediction import *
from threading import Thread
import threading
import cv2

red = 'color: red;'
green = 'color: green;'

StyleSheet = '''

QTabBar::tab:selected {
	/* expand/overlap to the left and right by 4px */
	margin-left: -4px;
	margin-right: -4px;
}

QTabBar::tab:first:selected {
	margin-left: 0; /* the first selected tab has nothing to overlap with on the left */
}

QTabBar::tab:last:selected {
	margin-right: 0; /* the last selected tab has nothing to overlap with on the right */
}

QTabBar::tab:only-one {
	margin: 0; /* if there is only one tab, we don't want overlapping margins */
}

QTabBar::tab{
	background-color:qlineargradient(spread:pad,x1:0, y1:0, x2:1, y2:0, 
													stop:0 rgba(245, 110, 32, 255), 
													stop:0.5 rgba(245, 208, 32,255), 
													stop:1 rgba(245, 110, 32, 255));

	color: rgb(0, 0, 255);
	border: 2px solid #C4C4C3;
	border-bottom-color: #C2C7CB; /* same as the pane color */
	border-top-left-radius: 8px;
	border-top-right-radius: 8px;
	min-width: 8ex;
	padding: 2px;
}
QTabBar::tab:!selected {
	margin-top: 5px; /* make non-selected tabs look smaller */
}
QTabBar::tab:selected, QTabBar::tab:hover {
	background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
								stop: 0 #fafafa, stop: 0.4 #f4f4f4,
								stop: 0.5 #e7e7e7, stop: 1.0 #fafafa);
}
'''
qtCreatorFile = "UI/trial_scancom_mutiltab.ui" # Enter file here.
Ui_MainWindow, _ = uic.loadUiType(qtCreatorFile ,
					from_imports=True,resource_suffix='', 
					import_from='UI')


class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
	def __init__(self):
		QtWidgets.QMainWindow.__init__(self)
		self.setWindowIcon(QtGui.QIcon('Images/vision.png'))
		Ui_MainWindow.__init__(self)
		self.setupUi(self)
		self.main_frame.setStyleSheet(StyleSheet)
		self.main_frame.tabBar().setCursor(QtCore.Qt.PointingHandCursor)

		#init parameters into gpu
		self.weights = init_weights()

		#for video capture,declare all cameras's parameters
		self._timer_side = QtCore.QTimer(self, interval=5)
		self._timer_side.timeout.connect(self._time_side)
		self._timer_side_1 = QtCore.QTimer(self, interval=5)
		self._timer_side_1.timeout.connect(self._time_side_1)
		self._timer_top = QtCore.QTimer(self, interval=5)
		self._timer_top.timeout.connect(self._time_top)

		self._maxCamerasToUse = 3
		self._tlFactory = pylon.TlFactory.GetInstance()
		self._devices = self._tlFactory.EnumerateDevices()
		self._cameras = pylon.InstantCameraArray(min(len(self._devices), self._maxCamerasToUse))

		self._camera_side = self._cameras[0]
		self._camera_side_1 = self._cameras[1]
		self._camera_top = self._cameras[2]

		self._camera_side.Attach(self._tlFactory.CreateDevice(self._devices[0]))
		self._camera_side_1.Attach(self._tlFactory.CreateDevice(self._devices[1]))
		self._camera_top.Attach(self._tlFactory.CreateDevice(self._devices[2]))

		self._camera_side.Open()
		self._camera_side_1.Open()
		self._camera_top.Open()


		self._camera_side.ExposureAuto.SetValue('Off')
		self._camera_side.ExposureTimeRaw.SetValue(3500)
		self._camera_side.Gamma.SetValue(2)
		self._camera_side.GammaEnable.SetValue(True)

		self._camera_side_1.ExposureAuto.SetValue('Off')
		self._camera_side_1.ExposureTimeRaw.SetValue(3500)
		self._camera_side_1.Gamma.SetValue(2)
		self._camera_side_1.GammaEnable.SetValue(True)

		self._camera_top.ExposureAuto.SetValue('Off')
		self._camera_top.ExposureTimeRaw.SetValue(3500)
		self._camera_top.GammaEnable.SetValue(False)


		self._cam_side = False
		self._cam_side_1 = False
		self._cam_top = False

		self._valid_product = True
		self._time_start = 10

		#start grabbing all cameras 
		self._trigger_side()
		self._trigger_side_1()
		self._trigger_top()
		self.pushButton.clicked.connect(self._start_testing)

	def _time_top(self):
		grabResult_top = self._camera_top.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
		if grabResult_top.GrabSucceeded():
			self._img_top = grabResult_top.Array
			self._img_top =  np.stack((self._img_top,)*3, axis=-1)
			self._img_top = cv2.resize(self._img_top,(self._img_top.shape[1]//2,self._img_top.shape[0]//2))
			temp = self._img_top.copy()			
			self._save_img_top = temp.copy()
			# self._img_top = cv2.resize(self._img_top,(640,375))
			h,w,_  = self._img_top.shape
			self._img_top = QImage(self._img_top,w, h,QImage.Format_RGB888)
			self.imageTop.setPixmap(QPixmap.fromImage(self._img_top))

	def _time_side(self):
		grabResult_side = self._camera_side.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
		if grabResult_side.GrabSucceeded():
			self._img_side = grabResult_side.Array
			self._img_side = np.stack((self._img_side,)*3, axis=-1)
			temp = self._img_side.copy()
			self._img_side = self._img_side[510:760,:]
			self._save_img_side = temp[510:760,:].copy()
			# self._img_side = cv2.resize(self._img_side,(self._img_side.shape[1]//2,self._img_side.shape[0]//2))
			h,w,_  = self._img_side.shape
			self._img_side = QImage(self._img_side,w, h,QImage.Format_RGB888)
			self.imageSide.setPixmap(QPixmap.fromImage(self._img_side))

	
	def _time_side_1(self):
    	
		grabResult_side_1 = self._camera_side_1.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
		# print(grabResult_side_1.GrabSucceeded())
		if grabResult_side_1.GrabSucceeded():
			self._img_side_1 = grabResult_side_1.Array
			self._img_side_1 = np.stack((self._img_side_1,)*3, axis=-1)
			temp = self._img_side_1.copy()
			self._img_side_1 = self._img_side_1[400:650,:]
			self._save_img_side_1 = temp[400:650,:].copy()
			# self._img_side_1 = cv2.resize(self._img_side,(self._img_side.shape[1]//2,self._img_side.shape[0]//2))
			h,w,_  = self._img_side_1.shape
			self._img_side_1 = QImage(self._img_side_1,w, h,QImage.Format_RGB888)
			self.imageSide_1.setPixmap(QPixmap.fromImage(self._img_side_1))


	def _trigger_side(self):
		if not self._timer_side.isActive():
			self._cam_side = True
			self._camera_side.StartGrabbing()
			self._timer_side.start(self._time_start)

	def _trigger_side_1(self):
		if not self._timer_side_1.isActive():
			self._cam_side_1 = True
			self._camera_side_1.StartGrabbing()
			self._timer_side_1.start(self._time_start)


	def _trigger_top(self):
		if not self._timer_top.isActive():
			self._cam_top = True
			self._camera_top.StartGrabbing()
			self._timer_top.start(self._time_start)


	# def _trigger_double(self):
	# 	self._double_cam = True
	# 	self._trigger_side()
	# 	self._trigger_top()

	
	def _start_testing(self):
		if self._cam_side and self._cam_top and self._cam_side_1:
			tic = time.time()
			self._camera_side.StopGrabbing()
			self._camera_side_1.StopGrabbing()
			self._camera_top.StopGrabbing()

			self.time.setText(time.ctime())
			c = predict(self._save_img_side,self._save_img_side_1,self._save_img_top,self.weights,mx.gpu(0))
			self._camera_side.StartGrabbing()
			self._camera_side_1.StartGrabbing()
			self._camera_top.StartGrabbing()
			self.latency.setText("{} giây".format(round(time.time()-tic,2)))

			side_items = [0]*4
			for item in c[0]:
				if item > 3 :
					continue
				side_items[item] += 1
			self._return_results([0,1,1,2])
			
			top_item = [0]*2
			for item in c[1]:	
				if item < 3 :
					continue
				top_item[item] += 1
			
			self._check_items(self.hardware,self.sttHardware,top_item[1],1)
			self._check_items(self.sole,self.sttSole,top_item[0],1)

			# self.hardware.setText(str(top_item[0]))
			# if top_item[0] == 1:				
			# 	self.sttHardware.setText("Đạt")
			# else:
			# 	self._valid_product = False
			# 	self.sttHardware.setText("Không Đạt")

			# self.hardware.setText(str(top_item[1]))
			# if top_item[1] == 1:
			# 	self._valid_product = False
			# 	self.hardware.setText("0")
			# 	self.stthardware.setText("Không Đạt") 

			if self._valid_product:
				self.sttProduct.setText("Đạt")
				self.sttProduct.setStyleSheet(green)
			else:
				self._valid_product = True
				self.sttProduct.setText("Không Đạt")
				self.sttProduct.setStyleSheet(red)


	def _check_items(self,item,sttItem,numbers,valid_num):
		item.setText(str(numbers))
		if numbers == valid_num:
			sttItem.setText("Đạt")
			sttItem.setStyleSheet(green)
			item.setStyleSheet(green)
		else:
			self._valid_product = False
			sttItem.setText("Không Đạt")
			sttItem.setStyleSheet(red)
			item.setStyleSheet(red)
			

	def _return_results(self,ls):
		self._check_items(self.backRight,self.sttBackRight,ls[2],1)
		self._check_items(self.backLeft,self.sttBackLeft,ls[1],1)
		self._check_items(self.frontLeg,self.sttFront,ls[3],2)

		# self.backRight.setText(str(ls[2]))
		# if ls[2] == 1 :
		# 	self.sttBackRight.setText('Đạt')
		# else:
		# 	self._valid_product = False
		# 	self.sttBackRight.setText('Không Đạt')

		# self.backLeft.setText(str(ls[1]))
		# if ls[1] == 1 :
		# 	self.sttBackLeft.setText('Đạt')
		# else:
		# 	self._valid_product = False
		# 	self.sttBackLeft.setText('Không Đạt')

		# self.frontLeg.setText(str(ls[3]))
		# if ls[3] == 2 :
		# 	self.sttFront.setText('Đạt')
		# else:
		# 	self._valid_product = False
		# 	self.sttFront.setText('Không Đạt')


if __name__ == "__main__":


	app = QtWidgets.QApplication(sys.argv)
	window = MyApp()
	window.show()
	sys.exit(app.exec_())
