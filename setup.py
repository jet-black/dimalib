#!/usr/bin/env python

from distutils.core import setup

setup(name='dimalib',
      version='1.0',
      description='My utils',
      author='jet-black',
      author_email='ovechkin.dm@gmail.com',
      url='https://github.com/jet-black/dimalib',
      packages=['dimalib'],
      install_requires=[
          'python_version>="3.4"',
          'tensorflow>=1.0.0',
          'Keras>=2.0.0',
          'matplotlib>=2.0.0',
          'numpy>=1.11.0'
      ])
