import os

os.environ["PYLON_CAMEMU"] = "3"

from pypylon import genicam
from pypylon import pylon
import sys
import cv2
import numpy as np

import time


maxCamerasToUse = 3

# The exit code of the sample application.
exitCode = 0

# Get the transport layer factory.
tlFactory = pylon.TlFactory.GetInstance()

# Get all attached devices and exit application if no device is found.
devices = tlFactory.EnumerateDevices()
if len(devices) == 0:
    raise pylon.RUNTIME_EXCEPTION("No camera present.")

cameras = pylon.InstantCameraArray(min(len(devices), maxCamerasToUse))
camera_side1 = cameras[0]
camera_side2 = cameras[1]
camera_top = cameras[2]

camera_side1.Attach(tlFactory.CreateDevice(devices[0]))
camera_side2.Attach(tlFactory.CreateDevice(devices[1]))
camera_top.Attach(tlFactory.CreateDevice(devices[2]))

camera_side1.Open()
camera_side2.Open()
camera_top.Open()

camera_side1.ExposureAuto.SetValue('Off')
camera_side1.ExposureTimeRaw.SetValue(3000)
camera_side1.Gamma.SetValue(2)
camera_side1.GammaEnable.SetValue(True)

camera_side2.ExposureAuto.SetValue('Off')
camera_side2.ExposureTimeRaw.SetValue(3000)
camera_side2.Gamma.SetValue(2)
camera_side2.GammaEnable.SetValue(True)

camera_top.ExposureAuto.SetValue('Off')
camera_top.ExposureTimeRaw.SetValue(3500)
# camera_side2.Gamma.SetValue(1)
camera_top.GammaEnable.SetValue(False)


camera_side2.StartGrabbing()
camera_side1.StartGrabbing()
camera_top.StartGrabbing()


while camera_side2.IsGrabbing() and camera_side1.IsGrabbing() and camera_top.IsGrabbing():
    grabResult_side1 = camera_side1.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
    grabResult_side2 = camera_side2.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
    grabResult_top = camera_top.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
    if grabResult_side1.GrabSucceeded() and grabResult_side2.GrabSucceeded() and grabResult_top.GrabSucceeded() :
        img_side1 = grabResult_side1.Array
        img_side2 =  grabResult_side2.Array
        img_top = grabResult_top.Array

        img_side1 = img_side1[400:650,:]
        img_side2 = img_side2[510:760,:]
        img_top = cv2.resize(img_top,(img_top.shape[1]//2,img_top.shape[0]//2))[500:,560:1760]

        # print(cv2.resize(img_side1,(img_side1.shape[1]//2,img_side1.shape[0]//2)).shape)
        # print(cv2.resize(img_side2,(img_side1.shape[1]//2,img_side2.shape[0]//2)).shape)
        # cv2.imshow("Side1",cv2.resize(img_side1,(img_side1.shape[1]//2,img_side1.shape[0]//2)))
        # cv2.imshow("Side2",cv2.resize(img_side2,(img_side1.shape[1]//2,img_side2.shape[0]//2)))
        cv2.imshow("Side1",img_side1)
        cv2.imshow("Side2",img_side2)
        cv2.imshow("Top",img_top)

        key = cv2.waitKey(30)
        if key == ord('q'):
            break

        if key == ord('c'):
            cv2.imwrite('debug/side1.jpg',img_side1)
            cv2.imwrite('debug/side2.jpg',img_side2)
            cv2.imwrite('debug/top.jpg',img_top)

        # break
    # grabResult_side1.Release()
    # grabResult_side2.Release()
    # grabResult_top.Release()
camera_side1.Close()
camera_side2.Close()

# Comment the following two lines to disable waiting on exit.
sys.exit(exitCode)