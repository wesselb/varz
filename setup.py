# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from setuptools import find_packages, setup

requirements = ['numpy>=1.16',
                'scipy<1.3',
                'autograd',
                'torch',
                'tensorflow',

                'plum-dispatch',
                'backends']

setup(packages=find_packages(exclude=['docs']),
      install_requires=requirements,
      include_package_data=True)