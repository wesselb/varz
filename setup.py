# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from setuptools import find_packages, setup

requirements = ['numpy>=1.16',
                'scipy>=1.3',

                'plum-dispatch',
                'backends>=0.3',
                'wbml']

setup(packages=find_packages(exclude=['docs']),
      python_requires='>=3.6',
      install_requires=requirements,
      include_package_data=True)
