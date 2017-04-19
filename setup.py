#!/usr/bin/env python

from setuptools import find_packages, setup

setup(name='dimalib',
      version='1.0',
      description='My utils',
      author='jet-black',
      author_email='ovechkin.dm@gmail.com',
      url='https://github.com/jet-black/dimalib',
      packages=find_packages(exclude=["*.test", "*.test.*"]),
      install_requires=[
          'tensorflow>=1.0.0',
          'numpy>=1.11.0'
      ])
