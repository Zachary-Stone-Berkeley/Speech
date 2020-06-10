import tensorflow as tf
import numpy as np
import os
from progress.bar import Bar
from tensorflow.python.tools.inspect_checkpoint import \
                                             print_tensors_in_checkpoint_file
# ---------------------------------
import layer_ops as ops
import init_helper

class AbstractClassifier:

  def __init__(self, sess=None,
                     data_set=None, 
                     batch_size=64, 
                     learning_rate=0.16, 
                     n_classes=2,  
                     name="classifier",
                     decay_op=lambda x: 1.,
                     n_features=None,
                     onehot_labels=True):
    
    if sess == None:
      config = tf.compat.v1.ConfigProto()
      config.gpu_options.allow_growth = True
      self.sess = tf.Session(config=config)
    else:
      self.sess = sess

    if not onehot_labels:
      assert n_classes == 2
    self.onehot = onehot_labels
    
    if data_set != None:
      self.DataSet_train, self.DataSet_test = data_set
      self.n_features = self.get_data('train').n_features
      print('\n\n', f'Num Input Features = {self.n_features}', '\n\n')
    else:
      assert n_features != None
      self.n_features = n_features

    self.batch_size = batch_size
    self.base_learning_rate = learning_rate
    self.learning_rate = tf.placeholder(shape=[], dtype=tf.float32)
    self.decay_op = decay_op

    self.name = name
    self.n_classes = n_classes

    self.build_architecture()
    self.build_loss()
    self.build_optimizers()
    if self.onehot:
      self.preds = tf.argmax(tf.nn.softmax(self.model), 1)
      self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.preds, 
        tf.argmax(self.label_placeholder, 1)), tf.float32))
    else:
      self.preds = tf.nn.relu(tf.math.sign(self.model))
      self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.preds, 
        self.label_placeholder), tf.float32))

    self.saver = tf.train.Saver(max_to_keep=1000)

  def build_architecture(self):
    pass

  def build_loss(self, loss_type='softmax'):
    if self.onehot:
      self.label_placeholder = tf.placeholder(shape=[None, self.n_classes], 
        dtype=tf.float32)
    else:
      self.label_placeholder = tf.placeholder(shape=[None, 1],  
        dtype=tf.float32)
    if loss_type == 'mse':
      self.loss = tf.reduce_mean(tf.keras.losses.MSE(self.label_placeholder, 
        self.model))
    elif loss_type == 'softmax':
      if self.onehot:
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
          labels=self.label_placeholder, logits=self.model))
      else:
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
          labels=self.label_placeholder, logits=self.model))

  def build_optimizers(self):
    with tf.variable_scope(self.name):
      optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, 
        momentum=0.9)
      self.gradients = optimizer.compute_gradients(self.loss, [var for var 
        in tf.trainable_variables() if self.name in var.name])
      self.update_op = optimizer.apply_gradients(self.gradients)

  def feed_dict(self, batch_size=None, shuffle=True, case='train'):
    '''
    returns a dictionary mapping input and label placeholders to the next 
    minibatch of inputs and labels (train)
    '''
    d = self.get_data(case)
    if batch_size == None:
      batch_size = self.batch_size
    minibatch_inputs, minibatch_labels = d.next_batch(
      batch_size, shuffle=shuffle)
    if self.onehot:
      if minibatch_labels.shape[1] == 1:
        t = np.zeros((batch_size, self.n_classes))
        minibatch_labels = t[np.arange(minibatch_labels.size), minibatch_labels] = 1
    else:
      if minibatch_labels.shape[1] > 1:
        minibatch_labels = np.reshape(np.argmax(minibatch_labels, axis=1), 
          (-1, 1))
    feed_dict = {self.input_placeholder: minibatch_inputs, 
                 self.label_placeholder: minibatch_labels,
                 self.learning_rate: self.base_learning_rate*self.decay}
    return feed_dict
  
  def get_data(self, case='train'):
    if case == 'train':
      return self.DataSet_train
    else:
      return self.DataSet_test

  def get_accuracy(self, case):
    nblocks = 1000
    acc = 0.
    n_examples = self.get_data(case).num_examples
    bl_size = n_examples//nblocks
    for i in range(nblocks):
      acc += self.sess.run([self.accuracy], feed_dict=self.feed_dict(case=case,
        batch_size=bl_size))[0]/nblocks
    return acc

  def predict(self, inputs):
    return self.sess.run(self.preds, feed_dict={
      self.input_placeholder:inputs})

  def get_errors(self, inputs, labels):
    return self.sess.run(tf.cast(tf.equal(
      tf.argmax(tf.nn.softmax(self.model), 1), 
      tf.argmax(self.label_placeholder, 1)), tf.int32), 
      feed_dict={self.input_placeholder:inputs, self.label_placeholder:labels})

  def train(self, num_epochs, test=False, close=True, save=True, save_path=''):
    '''
    num_epochs: the number of training epochs
    close: True or False. If true, will close the tensorflow session once done.
    '''
    num_train_minibatches = self.DataSet_train.num_examples//self.batch_size
    if test:
      num_test_minibatches = self.DataSet_test.num_examples//self.batch_size

    init_helper.initialize_uninitialized(self.sess)

    for epoch in range(num_epochs):
      
      self.decay = self.decay_op(epoch)

      bar = Bar(f'Training {self.name}: Epoch {epoch}', max=
        num_train_minibatches)      
      for step in range(num_train_minibatches):

        feed_dict = self.feed_dict()
        sess_result = self.sess.run([self.update_op], feed_dict=feed_dict)

        bar.next()
      bar.finish()
      

      train_acc = self.get_accuracy('train')      
      print('Train Accuracy: ', train_acc)
      if test:
        test_acc = self.get_accuracy('test')
        print('Test Accuracy: ', test_acc)
      print('----------')

    if save:
      if save_path:
        save_location = f'./pretrained models/{save_path}/'
      else:
        save_location = './pretrained models/'
      if not os.path.exists(save_location):
        os.makedirs(save_location)
      print(f'Saving weights to {save_location}')
      self.saver.save(self.sess, save_location + f'{self.name}.ckpt')

    if close:
      self.sess.close()

  def load_weights(self, weightfile):
    try:
      self.saver.restore(self.sess, weightfile)
    except Exception as e:
      print("Model could not load weights.\n")
      print("Weights in checkpoint:\n")
      print_tensors_in_checkpoint_file(weightfile, all_tensors=False, 
        all_tensor_names=True, tensor_name='')
      print("Weights in model:\n")
      for var in tf.trainable_variables():
        print(var.name)
      raise e

class AudioClassifier(AbstractClassifier):

  def __init__(self, n_neurons, pool=False, **kwargs):
    
    self.n_neurons = n_neurons
    self.pool = pool
    super(AudioClassifier, self).__init__(**kwargs)
  
  def build_architecture(self):
    
    self.input_placeholder = tf.placeholder(shape=[None, self.n_features], 
      dtype=tf.float32)
    output = self.input_placeholder

    output = ops.linear(output, self.n_neurons, f"{self.name}/dense")
    output = ops.activation_layer(output, tf.nn.relu)
    if self.onehot:
      output = ops.linear(output, self.n_classes, f"{self.name}/logits")
    else:
      output = ops.linear(output, 1, f"{self.name}/logits")
    self.model = output

class ComplexAudioClassifier(AbstractClassifier):

  def __init__(self, n_neurons, **kwargs):
    self.n_neurons = n_neurons
    super(ComplexAudioClassifier, self).__init__(**kwargs)
  
  def build_architecture(self):
    
    self.input_placeholder = tf.placeholder(shape=[None, self.n_features], 
      dtype=tf.float32)
    output = self.input_placeholder
    output = tf.expand_dims(output, 2)
    output = ops.conv1d(output, self.n_neurons, 5, 3, 0.02, 
      f"{self.name}/conv1d_01", padding="SAME")
    print(output.shape)
    output = ops.activation_layer(output, tf.nn.elu)
    #output = ops.max_pool_1d(output, size, stride)
    output = ops.flatten_layer(output)
    if self.onehot:
      output = ops.linear(output, self.n_classes, f"{self.name}/logits")
    else:
      output = ops.linear(output, 1, f"{self.name}/logits")
    self.model = output






