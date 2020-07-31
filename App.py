import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'
os.environ['MXNET_GPU_MEM_POOL_ROUND_LINEAR_CUTOFF'] = '26'
os.environ['MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD'] = '999'
os.environ['MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD'] = '25'
os.environ['MXNET_GPU_COPY_NTHREADS'] = '4'
os.environ['MXNET_OPTIMIZER_AGGREGATION_SIZE'] = '54'
import json
import sys
import threading
import time
import tkinter as tk
from shutil import copy
from threading import Thread
from tkinter import messagebox

import cv2
import mxnet as mx
import numpy as np
import PyQt5
from pypylon import pylon
from PyQt5 import QtCore, QtGui, QtWidgets, uic,QtSerialPort
from PyQt5.QtGui import QImage, QPixmap,QIcon
from PyQt5.QtWidgets import QMessageBox,QAction,QApplication
import UI.bg
from prediction import *
import PySimpleGUI as sg
import tkinter
import serial




red = 'color: red;'
green = 'color: green;'

bt_style = '''

border-style: outset;
border-width: 2px;

border-color: gray;
'''

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
	border-top-left-radius: 12px;
	border-top-right-radius: 12px;
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


class Process(QtCore.QThread):
	results  = QtCore.pyqtSignal(QImage)
	def __init__(self,images,type_img):
		QtCore.QThread.__init__(self)
		self._type = type_img
		self._img = images

	def run(self):
		if self._type == 'top':
			self._img = cv2.resize(self._img,(self._img.shape[1]//2,self._img.shape[0]//2))
		else :
			self._img = cv2.resize(self._img,(640,100))
		h,w,_  = self._img.shape
		self._img = QImage(self._img,w, h,QImage.Format_RGB888)
		self.results.emit(self._img)


# ser = serial.Serial('COM3', 9600,timeout = 1)
class SerialRead(QtCore.QThread):
	serialUpdate = QtCore.pyqtSignal(str)
	def run(self):
		while ser.is_open:
			# QThread.sleep(1)
			b = ser.readline()         # read a byte string
			string_n = b.decode()      # decode byte string into Unicode  
			string = string_n.rstrip()
			self.serialUpdate.emit(string)
			ser.flush()


class MyApp(QtWidgets.QMainWindow,Ui_MainWindow):
	def __init__(self):
		QtWidgets.QMainWindow.__init__(self)
		self.setWindowIcon(QtGui.QIcon('Images/vision.png'))
		Ui_MainWindow.__init__(self)
		self.setupUi(self)
		self.main_frame.setStyleSheet(StyleSheet)
		self.main_frame.tabBar().setCursor(QtCore.Qt.PointingHandCursor)

		self.serial = QtSerialPort.QSerialPort()
		self.serial.setPortName('COM3')
		self.serial.open(QtCore.QIODevice.ReadWrite)
		# self.setWindowFlag(QtCore.Qt.FramelessWindowHint)

		#init parameters into gpu		
		self.weights = init_weights()

		self._timer_wait = QtCore.QTimer(self, interval=5)
		self._timer_wait.timeout.connect(self._soft_trigger)

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

		self._classes = None
		self._cam_side = False
		self._cam_side_1 = False
		self._cam_top = False

		self._valid_product = True
		self._time_start = 20
		self._process = True
		self._render = False
		self._shot = 100
		self._button = 'UI/button.png'
		self.box = QMessageBox()
		self.box.setIcon(QMessageBox.Critical)
		self.box.setWindowIcon(QtGui.QIcon('UI/close.jpg'))

		# connect timeout signal to signal handler
		self.qTimer = QtCore.QTimer()
		self.qTimer.setInterval(100) # 1000 ms = 1 s		
		self.qTimer.timeout.connect(self._auto_render)
		self.qTimer.start()

		self._trigger_soft()

	def closeEvent(self, event):
		self.box.setWindowTitle('Thông báo')
		self.box.setText("Bạn có muốn thoát chương trình ?")
		self.box.setStandardButtons(QMessageBox.Yes|QMessageBox.No)

		buttonY = self.box.button(QMessageBox.Yes)
		buttonY.setText('Có')

		buttonN = self.box.button(QMessageBox.No)
		buttonN.setText('Không')
		self.box.exec_()

		if self.box.clickedButton() == buttonY:
			event.accept()
			# YES pressed
		elif self.box.clickedButton() == buttonN:
			# NO pressed
			event.ignore()


	def _auto_render(self):
		if self._render :
			s = (self.imageSide.geometry().width(),self.imageSide.geometry().height())
			s1 = (self.imageSide_1.geometry().width(),self.imageSide_1.geometry().height())
			s2 = (self.imageTop.geometry().width(),self.imageTop.geometry().height())
			for img,size,obj in zip([self._save_img_side,self._save_img_side_1,self._save_img_top],[s,s1,s2],[self.imageSide,self.imageSide_1,self.imageTop]):
				img = cv2.resize(img,size)
				h,w,_  = img.shape
				img = QImage(img,w, h,QImage.Format_RGB888)
				obj.setPixmap(QPixmap.fromImage(img))


	def render_img(self,img,type_img,obj):
		w0 = obj.geometry().width()
		h0 = obj.geometry().height()
		img = cv2.resize(img,(w0,h0))
		h,w,_  = img.shape
		img = QImage(img,w, h,QImage.Format_RGB888)
		obj.setPixmap(QPixmap.fromImage(img))
		self._render = False

	@QtCore.pyqtSlot(QImage)
	def render_side(self,img):
		self.imageSide.setPixmap(QPixmap.fromImage(img))


	@QtCore.pyqtSlot(QImage)
	def render_side_1(self,img):
		self.imageSide_1.setPixmap(QPixmap.fromImage(img))


	@QtCore.pyqtSlot(QImage)
	def render_top(self,img):
		self.imageTop.setPixmap(QPixmap.fromImage(img))


	def _time_top(self,save_folder):
		grabResult_top = self._camera_top.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
		if grabResult_top.GrabSucceeded():
			self._img_top = grabResult_top.Array
			cv2.imwrite(os.path.join(save_folder,'{}.jpg'.format(time.time())),self._img_top)
			self._img_top =  np.stack((self._img_top,)*3, axis=-1)
			self._img_top = cv2.resize(self._img_top,(self._img_top.shape[1]//2,self._img_top.shape[0]//2))
			temp = self._img_top.copy()
			self._save_img_top = temp.copy()
			self._camera_top.StopGrabbing()


	def _time_side(self,save_folder):
		grabResult_side = self._camera_side.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
		if grabResult_side.GrabSucceeded():
			self._img_side = grabResult_side.Array
			cv2.imwrite(os.path.join(save_folder,'{}.jpg'.format(time.time())),self._img_side)
			self._img_side = np.stack((self._img_side,)*3, axis=-1)
			temp = self._img_side.copy()
			self._img_side = self._img_side[400:650,:]
			self._save_img_side = temp[400:650,:].copy()
			self._camera_side.StopGrabbing()


	def _time_side_1(self,save_folder):
		grabResult_side_1 = self._camera_side_1.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
		if grabResult_side_1.GrabSucceeded():
			self._img_side_1 = grabResult_side_1.Array
			cv2.imwrite(os.path.join(save_folder,'{}.jpg'.format(time.time())),self._img_side_1)
			self._img_side_1 = np.stack((self._img_side_1,)*3, axis=-1)
			temp = self._img_side_1.copy()
			self._img_side_1 = self._img_side_1[500:750,:]
			self._save_img_side_1 = temp[500:750,:].copy()
			self._camera_side_1.StopGrabbing()

	def _trigger_side(self):
		if not self._timer_side.isActive() and self._process:
			self._cam_side = True
			self._timer_side.start(self._time_start)

	def _trigger_side_1(self):
		if not self._timer_side_1.isActive() and self._process:
			self._cam_side_1 = True
			self._timer_side_1.start(self._time_start)


	def _trigger_top(self):
		if not self._timer_top.isActive() and self._process:
			self._cam_top = True
			self._timer_top.start(self._time_start)


	def _hard_trigger(self):
		self._process = True

	
	def _trigger_soft(self):
		if not self._timer_wait.isActive() :
			self._timer_wait.start(self._time_start)


	def _soft_trigger(self):
		b = self.serial.readLine()
		string_n = b.data().decode() 
		string = string_n.rstrip() 		
		if string == '1':
			print(string)	
			time.sleep(0.1)		
			self._start_testing()


	def _start_testing(self):
		if self._process:
			name = 'img1/lan_{}'.format(self._shot)
			self._shot += 1
			os.mkdir(name)
			tic = time.time()

			self._camera_side.StartGrabbing()
			self._camera_side_1.StartGrabbing()
			self._camera_top.StartGrabbing()
			self._time_side(name)
			self._time_side_1(name)
			self._time_top(name)

			self.t = Thread(target=self.render_img, args=(self._save_img_side,1,self.imageSide,))
			self.t1 = Thread(target=self.render_img, args=(self._save_img_side_1,2,self.imageSide_1,))
			self.t2 = Thread(target=self.render_img, args=(self._save_img_top,'top',self.imageTop,))

			self.t.start()
			self.t1.start()
			self.t2.start()

			self.t.join()
			self.t1.join()
			self.t2.join()

			self.time.setText(time.ctime())
			c = predict(self._save_img_side,self._save_img_side_1,self._save_img_top[500:,560:1760],self.weights,mx.gpu(0),name)

			self._render = True
			self.latency.setText("{} giây".format(round(time.time()-tic,2)))
			items = [0]*6
			for item in c[0]:
				if item > 3 :
					continue
				items[item] += 1
			self._return_results(items)

			for item in c[1]:
				if item < 3 :
					continue
				items[item] += 1

			self._check_items(self.hardware,self.sttHardware,items[5],1)
			self._check_items(self.sole,self.sttSole,items[4],1)

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


if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	window = MyApp()
	window.show()
	sys.exit(app.exec_())
