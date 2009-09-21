#!/usr/bin/env python
''' Installation script for transforms3d package '''
from glob import glob
from distutils.core import setup

setup(name='transforms3d',
      version='0.1a',
      description='Semantic vector analysis package',
      author='Christoph Gohlke, Matthew Brett',
      author_email='Christoph Gohlke, matthew.brett@gmail.com',
      url='http://imaging.mrc-cbu.cam.ac.uk/svn/transforms3d',
      packages=['transforms3d'],
      scripts=glob('scripts/*.py')
      )

