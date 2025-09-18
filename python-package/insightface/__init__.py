# coding: utf-8
# pylint: disable=wrong-import-position
"""InsightFace: A Face Analysis Toolkit."""
from __future__ import absolute_import

try:
    #import mxnet as mx
    import onnxruntime
except ImportError:
    raise ImportError(
        "Unable to import dependency onnxruntime. "
    )

__version__ = '0.7.3'

from . import model_zoo
from . import utils
from . import app
from . import data
from . import thirdparty
from .utils.content_safety import ContentSafetyError

# Expose ContentSafetyError at package level for easy access
__all__ = ['model_zoo', 'utils', 'app', 'data', 'thirdparty', 'ContentSafetyError']

