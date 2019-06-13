# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from setuptools import find_packages, setup

requirements = ['numpy',
                'autograd',
                'scipy<=1.2.1',
                'torch',
                'tensorflow',

                'plum-dispatch',
                'backends']

setup(packages=find_packages(exclude=['docs']),
      install_requires=requirements)