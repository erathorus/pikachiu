import math

import tensorflow as tf
from tensorflow.keras.initializers import Initializer
from tensorflow.python.framework import dtypes


class StddevUniform(Initializer):
    def __init__(self, stddev, dtype=dtypes.float32):
        self.stddev = stddev
        self.dtype = dtypes.as_dtype(dtype)

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        return tf.random_uniform(
            shape=shape,
            minval=-self.stddev * math.sqrt(3),
            maxval=self.stddev * math.sqrt(3)
        )

    def get_config(self):
        return {"dtype": self.dtype.name}
