import os

os.environ["PYLON_CAMEMU"] = "3"

from pypylon import genicam
from pypylon import pylon
import sys
import cv2
import numpy as np

import time
# Number of images to be grabbed.
countOfImagesToGrab = 10

# Limits the amount of cameras used for grabbing.
# It is important to manage the available bandwidth when grabbing with multiple cameras.
# This applies, for instance, if two GigE cameras are connected to the same network adapter via a switch.
# To manage the bandwidth, the GevSCPD interpacket delay parameter and the GevSCFTD transmission delay
# parameter can be set for each GigE camera device.
# The "Controlling Packet Transmission Timing with the Interpacket and Frame Transmission Delays on Basler GigE Vision Cameras"
# Application Notes (AW000649xx000)
# provide more information about this topic.
# The bandwidth used by a FireWire camera device can be limited by adjusting the packet size.
maxCamerasToUse = 2

# The exit code of the sample application.
exitCode = 0



    # Get the transport layer factory.
tlFactory = pylon.TlFactory.GetInstance()

# Get all attached devices and exit application if no device is found.
devices = tlFactory.EnumerateDevices()
if len(devices) == 0:
    raise pylon.RUNTIME_EXCEPTION("No camera present.")

cameras = pylon.InstantCameraArray(min(len(devices), maxCamerasToUse))
camera_side = cameras[0]
camera_top = cameras[1]
camera_side.Attach(tlFactory.CreateDevice(devices[0]))
camera_top.Attach(tlFactory.CreateDevice(devices[1]))
camera_side.Open()
camera_top.Open()

camera_side.ExposureAuto.SetValue('Off')
camera_side.ExposureTimeRaw.SetValue(30000)
camera_side.Gamma.SetValue(2)
camera_side.GammaEnable.SetValue(True)

camera_top.ExposureAuto.SetValue('Off')
camera_top.ExposureTimeRaw.SetValue(105000)
camera_top.GammaEnable.SetValue(False)
numberOfImagesToGrab = 1000
camera_top.StartGrabbing()
camera_side.StartGrabbing()
while camera_top.IsGrabbing() and camera_side.IsGrabbing():
    grabResult_side = camera_side.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
    grabResult_top = camera_top.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if grabResult_side.GrabSucceeded():
        img_side = grabResult_side.Array
        img_top =  grabResult_top.Array
        img_side = img_side[300:500,:]
        img_top = cv2.resize(img_top,(img_top.shape[1]//2,img_top.shape[0]//2))[200:950,:1200]
        # print(cv2.resize(img_side,(img_side.shape[1]//2,img_side.shape[0]//2)).shape)
        # print(cv2.resize(img_top,(img_side.shape[1]//2,img_top.shape[0]//2)).shape)
        cv2.imshow("Side",cv2.resize(img_side,(img_side.shape[1]//2,img_side.shape[0]//2)))
        cv2.imshow("Top",cv2.resize(img_top,(img_side.shape[1]//2,img_top.shape[0]//2)))
        key = cv2.waitKey(30)
        if key == ord('q'):
            break
        # break
    grabResult_side.Release()
camera_side.Close()
camera_top.Close()

# Comment the following two lines to disable waiting on exit.
sys.exit(exitCode)