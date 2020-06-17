import os

os.environ["PYLON_CAMEMU"] = "3"

from pypylon import genicam
from pypylon import pylon
import sys
import cv2
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

try:

    # Get the transport layer factory.
    tlFactory = pylon.TlFactory.GetInstance()

    # Get all attached devices and exit application if no device is found.
    devices = tlFactory.EnumerateDevices()
    if len(devices) == 0:
        raise pylon.RUNTIME_EXCEPTION("No camera present.")

    cameras = pylon.InstantCameraArray(min(len(devices), maxCamerasToUse))

    for i, camera in enumerate(cameras):
        camera.Attach(tlFactory.CreateDevice(devices[i]))
        # print("Using device ", camera.GetDeviceInfo().GetModelName())
        camera.Open()
        if camera.GetDeviceInfo().GetModelName() == "acA1300-30gm":
            camera.ExposureAuto.SetValue('Off')
            camera.ExposureTimeRaw.SetValue(3000)
            camera.Gamma.SetValue(2)
            camera.GammaEnable.SetValue(True)
        else :
            camera.ExposureAuto.SetValue('Off')
            camera.ExposureTimeRaw.SetValue(35000)
            camera.GammaEnable.SetValue(False)
        numberOfImagesToGrab = 1000
        camera.StartGrabbingMax(numberOfImagesToGrab)

        while camera.IsGrabbing():
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if grabResult.GrabSucceeded():
                img = grabResult.Array
                if camera.GetDeviceInfo().GetModelName() == 'acA1300-30gm':
                    img = img[300:500,:]
                else:
                    img = img[600:1700,:]
                cv2.imshow("out",img)
                key = cv2.waitKey(30)
                if key == ord('q'):
                    break
                if key == ord('c'):
                    cv2.imwrite("{}.jpg".format(time.time()),img)
            grabResult.Release()
        camera.Close()
        del camera

except genicam.GenericException as e:
    # Error handling
    print("An exception occurred.", e.GetDescription())
    exitCode = 1

# Comment the following two lines to disable waiting on exit.
sys.exit(exitCode)