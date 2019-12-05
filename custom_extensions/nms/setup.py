"""
Created at 07.11.19 19:12
@author: gregor

"""

import os, sys
from pathlib import Path

from setuptools import setup
from torch.utils import cpp_extension

dir_ = Path(os.path.dirname(sys.argv[0]))

setup(name='nms_extension',
      ext_modules=[cpp_extension.CUDAExtension('nms_extension', [str(dir_/'src/nms_interface.cpp'), str(dir_/'src/nms.cu')])],
      cmdclass={'build_ext': cpp_extension.BuildExtension}
      )