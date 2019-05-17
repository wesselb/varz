# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..')))

# noinspection PyUnresolvedReferences
from varz import *

from .util import *

# Load LAB extensions.
# noinspection PyUnresolvedReferences
import lab.tensorflow
# noinspection PyUnresolvedReferences
import lab.torch
