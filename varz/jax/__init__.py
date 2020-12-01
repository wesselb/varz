import jax

# noinspection PyUnresolvedReferences
import lab.jax

from .minimise import *

# noinspection PyUnresolvedReferences
from .. import *

# We will need `float64`s
jax.config.update("jax_enable_x64", True)
