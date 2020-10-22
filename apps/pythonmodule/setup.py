import os
import re
import sys
import platform
from os.path import exists, join
import shutil

from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

# The following variables are configure by CMake
walberla_source_dir = "${walberla_SOURCE_DIR}"
walberla_binary_dir = "${CMAKE_CURRENT_BINARY_DIR}"

if platform.system() == 'Windows':
    extension = ('dll', 'pyd')
    configuration = 'Release'
else:
    extension = ('so', 'so')
    configuration = ''

src_shared_lib = join(walberla_binary_dir, configuration, 'walberla_cpp.' + extension[0])


packages = ['waLBerla']


setup(
    name='waLBerla',
    version='1.0',
    author='Markus Holzer',
    author_email='markus.holzer@fau.de',
    url='http://www.walberla.net',
    description='waLBerla python bindings with pybind11',
    long_description='',
    packages=find_packages(src_shared_lib),
)
