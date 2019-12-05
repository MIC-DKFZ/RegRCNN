"""
Created at 07.11.19 19:12
@author: gregor

"""


from setuptools import setup
from torch.utils import cpp_extension

setup(name='sandbox_cuda',
      ext_modules=[cpp_extension.CUDAExtension('sandbox', ['src/sandbox.cpp', 'src/sandbox_cuda.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})