# contents of setup.py
# from cx_Freeze import setup, Executable

# import distutils
# import opcode
# import os

# # opcode is not a virtualenv module, so we can use it to find the stdlib; this is the same
# # trick used by distutils itself it installs itself into the virtualenv
# distutils_path = os.path.join(os.path.dirname(opcode.__file__), 'distutils')
# build_exe_options = {'include_files': [(distutils_path, 'distutils')], "excludes": ["distutils"]}

# setup(
#     name="App",
#     version="0.1",
#     description="My app",
#     options={"build_exe": build_exe_options},
#     executables=[Executable("barcode_scanner_video.py", base=None)],
# )

"""
File based on a contribution from Josh Immanuel. Use via 

python setup-py2exe.py py2exe

which will create a dist folder containing the .exe, the python DLL, and a 
few other DLL deemed by py2exe to be critical to the application execution. 

The contents of the dist folder should then be packaged using a tool such 
as NSIS or Inno Setup. The py2exe page has an example for NSIS. 
"""

#setup.py
import sys, os
import time
from cx_Freeze import setup, Executable

__version__ = "1.1.0"

include_files = []
excludes =[]
# excludes = ["tkinter","pandas","imutils","matplotlib","flask","flask-cors",]
# packages = ['cv2','time','pypylon','pyzbar','numpy','PyQt5'] 
packages = []
setup(
    name = "My app",
    description='App Description',
    version=__version__,
    options = {"build_exe": {
    'packages': packages,
    'include_files': include_files,
    'excludes': excludes,
    'include_msvcr': True,
}},
executables = [Executable("App.py",base=None)]
)
