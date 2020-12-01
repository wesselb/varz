import os
import sys

# Add package to path.
file_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(file_dir, "..")))

# Load LAB extensions.
# noinspection PyUnresolvedReferences
import lab.torch

# noinspection PyUnresolvedReferences
import lab.tensorflow

# Load TensorFlow extension.
# noinspection PyUnresolvedReferences
import varz.tensorflow
