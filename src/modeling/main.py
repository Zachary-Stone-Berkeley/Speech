import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
# ---------------------------------
from DataGetter import get_data
from CustomDataSet import DataSet
from models import AudioClassifier, ComplexAudioClassifier

tf.logging.set_verbosity(tf.logging.ERROR)

if __name__ == "__main__":

  print(tf.__version__)

  def str2bool(x):
    if isinstance(x, bool):
      return x
    if x.lower() == 'true':
      return True
    elif x.lower() == 'false':
      return False
    else:
      raise ValueError("Argument must be \"true\" or \"false\".")

  parser = argparse.ArgumentParser()

  # arguments related to training
  parser.add_argument('-bs', '--batch_size', type=int, default=64, 
    help="Integer batch size for training.")
  parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, 
    help="Float learning rate for training.")
  parser.add_argument('-e', '--num_epochs', type=int, default=5, 
    help="Integer number of epochs to train for.")
  parser.add_argument('-nn', '--n_neurons', type=int, default=8, 
    help="Integer number of neurons in the dense layer or output channels in \
    the conv layer.")

  # argument related to the data
  parser.add_argument('-r', '--reuse', type=str2bool, default=False,
    help="If True, sources will not be combined to make the data set, instead \
    the existing combined csv will be reused if it exists.")
  parser.add_argument('-gb', '--max_gb', type=float, default=1.,
    help="Maximum gb csv file.")
  #parser.add_argument('--speech_src', type=str, default=None,
  #  help="The speech source to be used.")
  #parser.add_argument('--nonspeech_src', type=str, default=None,
  #  help="The non-speech source to be used.")
  parser.add_argument('--sources', nargs='+', type=str,
    default=['amicorpus', 'DCASE', 'fma_medium'])
  parser.add_argument('--mem_friendly', type=str2bool, default=False,
    help="Set to true -gb is set to more than can be loaded into \
    memory.")
  parser.add_argument('--path', type=str, default='',
    help='Path from root/datadir/ to data set.')
  parser.add_argument('--name', type=str, default='',
    help='Name for model (used in saving/loading weights).')
  parser.add_argument('--scale', type=str2bool, default=False,
    help='Scale the data.')

  # argument related to the model
  parser.add_argument('--test', type=str2bool, default=True, 
    help="Bool to compute test accuracy or not.")
  parser.add_argument('--conv', type=str2bool, default=False, 
    help="Bool to use convolution or not.")  
  parser.add_argument('-s', '--save', type=str2bool, default=True,
    help="Bool to save model weights or not.")
  parser.add_argument('-oh', '--onehot', type=str2bool, default=True,
    help="One-hot or not.")
  parser.add_argument('-p', '--pool', type=str2bool, default=False,
    help='Experimental pooling in feed forward network.')

  args = parser.parse_args()

  config = tf.compat.v1.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)

  if not args.name:
    if args.conv:
      name = 'conv'
    else:
      name = 'feedforward'
  else:
    name = args.name
  
  print('\nLoading data set...\n')
  reuse = args.reuse
  classmap = [x for i, x in enumerate(['nonspeech', 'speech', 'music']) 
                                                           if args.sources[i]]
  n_classes = len(classmap)
  d = get_data(sources=args.sources, classmap=classmap,
    max_gb=args.max_gb, reuse=reuse, memory_friendly=args.mem_friendly,
    data_path=args.path, doscale=args.scale)
  
  print(f'\nBuilding model named {name}...\n')
  if args.conv:
    decay_op = lambda x: 10**(-x//10) 
    c = ComplexAudioClassifier(n_neurons=args.n_neurons, sess=sess, data_set=d, 
      batch_size=args.batch_size, learning_rate=args.learning_rate, 
      n_classes=n_classes, name=name, decay_op=decay_op, 
      onehot_labels=args.onehot)
  else:
    decay_op = lambda x: 1.
    c = AudioClassifier(n_neurons=args.n_neurons, sess=sess, data_set=d, 
      batch_size=args.batch_size, learning_rate=args.learning_rate, 
      n_classes=n_classes, name=name, decay_op=decay_op, 
      onehot_labels=args.onehot, pool=args.pool)
  
  print(f'\nTraining model for {args.num_epochs} epochs...\n')
  c.train(num_epochs=args.num_epochs, test=args.test, close=False, 
    save=args.save, save_path=args.path)
  if not args.mem_friendly:
    ns = []
    for i in range(c.n_classes):
      n = np.sum(np.argmax(c.get_data().list_of_data[1], axis=1) == i)
      ns.append(n)
      print(f'Num {classmap[i]} = {n}')
    print('Best guess accuracy = ', max(ns)/np.sum(ns))
  
  if not args.conv and not args.pool:
    for v in tf.trainable_variables():
      if c.name in v.name:
        npv = sess.run(v)
        name = v.name.replace('/', '.')
        if args.path:
          path_to_weights = './weights/' + args.path
        else:
          path_to_weights = './weights'
        if not os.path.exists(path_to_weights):
          os.makedirs(path_to_weights)
        with open(path_to_weights + f'/{name}.txt', 'w+') as f:
          if len(npv.shape) > 1:
            for row in np.transpose(npv):
              f.write(','.join(str(x) for x in row) + '\n')
          else:
            for row in npv:
              f.write(str(row) + '\n')


  sess.close()