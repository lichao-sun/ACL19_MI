""" Tool functions are provided here.
"""

import numpy as np
import tensorflow as tf

def norm_matrix(matrix):
      """ normalize matrix by column
	    input : numpy array, dtype = float32
	    output : normalized numpy array, dtype = float32
      """
      # check dtype of the input matrix
      np.testing.assert_equal(type(matrix).__name__, 'ndarray')
      np.testing.assert_equal(matrix.dtype, np.float32)

      row_sums = matrix.sum(axis=1)
      # replace zero denominator
      row_sums[row_sums == 0] = 1
      norm_matrix = matrix / row_sums[:, np.newaxis]
      return norm_matrix

def sample_gumbel(shape, eps=1e-10):
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape,minval=0,maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax( y / temperature)

def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        k = tf.shape(logits)[-1]
        #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
        y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y
