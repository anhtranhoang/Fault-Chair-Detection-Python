import json
import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'
os.environ['MXNET_GPU_MEM_POOL_ROUND_LINEAR_CUTOFF'] = '26'
os.environ['MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD'] = '999'
os.environ['MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD'] = '25'
os.environ['MXNET_GPU_COPY_NTHREADS'] = '4'
os.environ['MXNET_OPTIMIZER_AGGREGATION_SIZE'] = '54'

import sys
import time
from shutil import copy

import cv2
import mxnet as mx
import numpy as np
from pypylon import pylon
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtGui import QImage, QPixmap
from prediction import *
from threading import Thread
import threading
import cv2

qtCreatorFile = "UI/demo.ui" # Enter file here.
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)




class MyApp(QtWidgets.QMainWindow, Ui_MainWindow,threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)
		QtWidgets.QMainWindow.__init__(self)
		self.setWindowIcon(QtGui.QIcon('Images/vision.png'))
		Ui_MainWindow.__init__(self)
		self.setupUi(self)
		self.weights = init_weights()

		#for video capture
		self._timer_side = QtCore.QTimer(self, interval=5)
		self._timer_side.timeout.connect(self._time_side)
		self._timer_top = QtCore.QTimer(self, interval=5)
		self._timer_top.timeout.connect(self._time_top)

		#button trigger
		self.triggerSide.clicked.connect(self._trigger_side)
		self.triggerTop.clicked.connect(self._trigger_top)
		self.triggerDouble.clicked.connect(self._trigger_double)
		self.startTesting.clicked.connect(self._start_testing)



		self._maxCamerasToUse = 2
		self._tlFactory = pylon.TlFactory.GetInstance()
		self._devices = self._tlFactory.EnumerateDevices()
		self._cameras = pylon.InstantCameraArray(min(len(self._devices), self._maxCamerasToUse))
		self._camera_side = self._cameras[0]
		self._camera_top = self._cameras[1]
		self._camera_side.Attach(self._tlFactory.CreateDevice(self._devices[0]))
		self._camera_top.Attach(self._tlFactory.CreateDevice(self._devices[1]))
		self._camera_side.Open()
		self._camera_top.Open()

		self._camera_side.ExposureAuto.SetValue('Off')
		self._camera_side.ExposureTimeRaw.SetValue(2500)
		self._camera_side.Gamma.SetValue(2)
		self._camera_side.GammaEnable.SetValue(True)

		self._camera_top.ExposureAuto.SetValue('Off')
		self._camera_top.ExposureTimeRaw.SetValue(28000)
		self._camera_top.DigitalShift.SetValue(1)
		self._camera_top.GammaEnable.SetValue(False)
		self.threadpool = QtCore.QThreadPool()
		self._cam_side = False
		self._cam_top = False
		self._valid_product = True
		self._double_cam = False
		self._check_top = False
		self._is_waiting = True
		self._img = None
		self._start = False


	def _time_top(self):
		grabResult_top = self._camera_top.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
		if grabResult_top.GrabSucceeded():
			self._img_top = grabResult_top.Array
			self._img_top =  np.stack((self._img_top,)*3, axis=-1)
			self._img_top = cv2.resize(self._img_top,(self._img_top.shape[1]//2,self._img_top.shape[0]//2))[200:950,:1200]
			temp = self._img_top.copy()			
			self._save_img_top = temp.copy()
			
			cv2.rectangle(self._img_top,(140,40),(1280,410),(255,0,0),2)
			self._img_top = cv2.resize(self._img_top,(640,375))
			h,w,_  = self._img_top.shape
			self._img_top = QImage(self._img_top,w, h,QImage.Format_RGB888)
			self.imageTop.setPixmap(QPixmap.fromImage(self._img_top))
			print(1)

	def _time_side(self):
		grabResult_side = self._camera_side.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
		if grabResult_side.GrabSucceeded():
			self._img_side = grabResult_side.Array
			self._img_side = np.stack((self._img_side,)*3, axis=-1)
			temp = self._img_side.copy()
			for x in [220,450,890,1100]:				
				cv2.line(self._img_side,(x,0),(x,self._img_side.shape[0]),(255,0,0),2)
			self._img_side = self._img_side[330:530,:]
			self._save_img_side = temp [330:530,:].copy()
			self._img_side = cv2.resize(self._img_side,(self._img_side.shape[1]//2,self._img_side.shape[0]//2))
			h,w,_  = self._img_side.shape
			self._img_side = QImage(self._img_side,w, h,QImage.Format_RGB888)
			self.imageSide.setPixmap(QPixmap.fromImage(self._img_side))


	def _trigger_side(self):
		if not self._timer_side.isActive():
			self._cam_side = True
			self._double_cam = False
			self._camera_side.StartGrabbing()
			self._timer_side.start(20)

	def _trigger_top(self):
		if not self._timer_top.isActive():
			self._cam_top = True
			self._double_cam = False
			self._camera_top.StartGrabbing()
			self._timer_top.start(20)


	def _trigger_double(self):
		self._double_cam = True
		self._trigger_side()
		self._trigger_top()

	
	def _start_testing(self):
		if self._cam_side and self._cam_top:
			tic = time.time()
			self._camera_side.StopGrabbing()
			self._camera_top.StopGrabbing()

			# self._save_img_side = np.stack((self._save_img_side,)*3, axis=-1)
			# self._save_img_top = np.stack((self._save_img_top,)*3, axis=-1)

			self.time.setText(time.ctime())
			c,self._check_top = predict(self._save_img_side,self._save_img_top,self.weights,mx.gpu(0))
			self._camera_side.StartGrabbing()
			self._camera_top.StartGrabbing()
			self.latency.setText("{} giây".format(round(time.time()-tic,2)))
			side_items = [0]*4
			for item in c[0]:
				if item > 3 :
					continue
				side_items[item] += 1
			self._return_results(side_items)

			if len(c[1]):
				if self._check_top:
					self.joint.setText("1")
					self.sttJoint.setText("Đạt")
				else:
					self._valid_product = False
					self.joint.setText("0")
					self.sttJoint.setText("Không Đạt")
			else:
				self._valid_product = False
				self.joint.setText("0")
				self.sttJoint.setText("Không Đạt")

			if self._valid_product:
				self.sttProduct.setText("Đạt")
			else:
				self._valid_product = True
				self.sttProduct.setText("Không Đạt")




	def _return_results(self,ls):

		self.backRight.setText(str(ls[2]))
		if ls[2] == 1 :
			self.sttBackRight.setText('Đạt')
		else:
			self._valid_product = False
			self.sttBackRight.setText('Không Đạt')

		self.backLeft.setText(str(ls[1]))
		if ls[1] == 1 :
			self.sttBackLeft.setText('Đạt')
		else:
			self._valid_product = False
			self.sttBackLeft.setText('Không Đạt')

		self.frontLeg.setText(str(ls[3]))
		if ls[3] == 2 :
			self.sttFront.setText('Đạt')
		else:
			self._valid_product = False
			self.sttFront.setText('Không Đạt')


if __name__ == "__main__":


	app = QtWidgets.QApplication(sys.argv)
	window = MyApp()
	window.show()
	sys.exit(app.exec_())
