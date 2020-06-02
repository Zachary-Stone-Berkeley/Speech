import tensorflow as tf

def weight_initializer(initializer="glorot_normal", stddev=0.02):
  """Returns the initializer for the given name.

  Args:
    initializer: Name of the initalizer. Use one in consts.INITIALIZERS.
    stddev: Standard deviation passed to initalizer.

  Returns:
    Initializer from `tf.initializers`.
  """
  if initializer == "normal":
    return tf.initializers.random_normal(stddev=stddev)
  if initializer == "truncated":
    return tf.initializers.truncated_normal(stddev=stddev)
  if initializer == "orthogonal":
    return tf.initializers.orthogonal()
  if initializer == "glorot_normal":
    return tf.initializers.glorot_normal()
  raise ValueError("Unknown weight initializer {}.".format(initializer))

def activation_layer(inputs, activation):
  return activation(inputs)

def flatten_layer(inputs):
  return tf.contrib.layers.flatten(inputs)

def max_pool(inputs, size, stride, padding='SAME', name='max_pool'):
  output = tf.nn.max_pool(inputs, size, stride, padding, 'NHWC', name)
  return output

def linear(inputs, output_size, scope=None, stddev=0.02, bias_start=0.0,
           use_sn=False, use_bias=True):
  """Linear layer without the non-linear activation applied."""
  shape = inputs.get_shape().as_list()
  with tf.variable_scope(scope or "linear", reuse=tf.AUTO_REUSE):
    kernel = tf.get_variable(
        "kernel",
        [shape[1], output_size],
        initializer=weight_initializer())
    if use_sn:
      kernel = spectral_norm(kernel)
    outputs = tf.matmul(inputs, kernel)
    if use_bias:
      bias = tf.get_variable(
          "bias",
          [output_size],
          initializer=weight_initializer())#tf.constant_initializer(0.0))
      outputs += bias
    return outputs

def conv2d(inputs, output_dim, k_h, k_w, d_h, d_w, stddev=0.02, name="conv2d",
           use_sn=False, use_bias=True, padding="SAME"):
  """Performs 2D convolution of the input."""
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    w = tf.get_variable(
        "kernel", [k_h, k_w, inputs.shape[-1].value, output_dim],
        initializer=weight_initializer())
    outputs = tf.nn.conv2d(inputs, w, strides=[1, d_h, d_w, 1], padding=padding)
    if use_bias:
      bias = tf.get_variable(
          "bias", [output_dim], initializer=tf.constant_initializer(0.0))
      outputs += bias
  return outputs

def conv1d(inputs, output_dim, size, stride, stddev=0.02, name="conv1d", 
           use_sn=False, use_bias=True, padding="SAME"):
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    w = tf.get_variable("kernel", [size, inputs.shape[-1].value, output_dim], 
      initializer=weight_initializer())
    outputs = tf.nn.conv1d(inputs, w, stride, padding=padding)
    if use_bias:
      bias = tf.get_variable(
          "bias", [output_dim], initializer=tf.constant_initializer(0.0))
      outputs += bias
  return outputs

def max_pool_1d_DEPRECATED(inputs, size, stride, padding='SAME', 
                           name='max_pool'):
  if inputs == None:
    return lambda x: max_pool_1d(x, size, stride, padding, name)
  else:
    output = tf.nn.max_pool1d(inputs, size, stride, padding, 'NHWC', name)
    return output

def max_pool_1d(inputs, size, stride, padding='SAME', name='max_pool'):
  if inputs == None:
    return lambda x: max_pool_1d(x, size, stride, padding, name)
  else:
    assert len(inputs.shape) in [2, 3, 4]
    output = inputs
    if len(output.shape) == 2:
      output = tf.expand_dims(inputs, 2)
    if len(output.shape) == 3:
      output = tf.expand_dims(output, 1)
    #output = tf.expand_dims(inputs, 1) if len(inputs.shape) == 3 else inputs
    output = max_pool(output, size, stride, padding, name)
    return output

def max_pool(inputs, size, stride, padding='SAME', name='max_pool'):
  if inputs == None:
    return lambda x: max_pool(x, size, stride, padding, name)
  else:
    output = tf.nn.max_pool(inputs, size, stride, padding, 'NHWC', name)
    return output