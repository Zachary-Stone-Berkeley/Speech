import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import argparse
import numpy as np
import librosa
import pyaudio
from functools import reduce
from progress.bar import Bar
from sklearn.preprocessing import scale
# ------
from DataGetter import load_csv
from models import AudioClassifier, ComplexAudioClassifier
from feature_extraction import xtract, classify, xtractv2, classify2
from tf_model import tf_classify
from scaling import mu, var

SCALE = True
def myscale(x):
  return (x-mu)/var

tf.logging.set_verbosity(tf.logging.ERROR)

EXTENSIONS = ['.wav', '.mp3', '.flac']

def get_paths(source):
  root = f'../../{source}/'
  paths = [[x[0] + '/' + y for y in x[2] if 
    any(extension in y for extension in EXTENSIONS)] for 
      x in os.walk(root)]
  paths = reduce((lambda a,b: a+b), paths)
  np.random.shuffle(paths)
  return paths

def str2bool(x):
  if isinstance(x, bool):
    return x
  if x.lower() == 'true':
    return True
  elif x.lower() == 'false':
    return False
  else:
    raise ValueError("Argument must be \"true\" or \"false\".")

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('-nn', '--n_neurons', type=int, default=8, 
    help="number of neurons")
  parser.add_argument('-nf', '--n_features', type=int, default=95,
    help="number of input features")
  parser.add_argument('-oh', '--onehot', type=str2bool, default=False)
  parser.add_argument('-tf', '--tensorflow', type=str2bool, default=False)
  parser.add_argument('--conv', type=str2bool, default=False)
  parser.add_argument('-p', '--pool', type=str2bool, default=False)
  parser.add_argument('--path', type=str, default='')
  parser.add_argument('--name', type=str, default='')
  args = parser.parse_args()

  if not args.name:
    if args.conv:
      name = 'conv'
    else:
      name = 'feedforward'
  else:
    name = args.name

  
  if args.n_features == 95:
    XTRACT = xtract
    ALPHA = 0.01
    BETA = 0.01
    N_INTERLEAVINGS = 5
  elif args.n_features == 76:
    XTRACT = xtractv2
    ALPHA = 0.015
    BETA = 0.015
    N_INTERLEAVINGS = 3
  elif args.n_features == 114:
    XTRACT = xtractv2
    ALPHA = 0.01
    BETA = 0.01
    N_INTERLEAVINGS = 5
  else:
    raise Exception

  if args.tensorflow:
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    if not args.conv:
      c = AudioClassifier(n_neurons=args.n_neurons, sess=sess, 
        n_features=args.n_features, n_classes=2, name=name,
        onehot_labels=args.onehot, pool=args.pool)
    else:
      c = ComplexAudioClassifier(n_neurons=args.n_neurons, sess=sess, 
        n_features=args.n_features, n_classes=2, name=name,
        onehot_labels=args.onehot)
    if args.path:
      c.load_weights(f'./pretrained models/{args.path}/{c.name}.ckpt')
    else:
      c.load_weights(f'./pretrained models/{c.name}.ckpt')
    
    def tfclassify(x):
      if x.shape[0] == -1:
        return c.predict(x)[0]
      else:
        return c.predict(x)
  
  else:
    tfclassify = tf_classify

  do = input('Test on other speech data sets? y/n\n')
  if do == 'y':
    size = 100
    for dataset in ('LibriSpeech', 'Vox Forge', 'Flickr Audio Caption'):
      paths = get_paths(dataset)
      np.random.shuffle(paths)
      bar = Bar(dataset, max=size)
      n_disagreements = 0
      co_errors = 0
      mo_errors = 0
      n_instances = 0
      
      for wav in paths[:size]:
        bar.next()
        audio, sr = librosa.load(wav, sr=None)
        if len(audio) < 0.5:
          continue
        xf = XTRACT(audio, sr, alpha=ALPHA, beta=BETA, 
        	n_interleavings=N_INTERLEAVINGS)
        if SCALE:
          xf = myscale(xf)
        for i, xf in enumerate(xf):
          n_instances += 1
          co = 1 #if classify2(xf) else 0
          if args.tensorflow:
            mo = tfclassify(np.reshape(xf, (1, args.n_features)))
          else:
            mo = tfclassify(xf)
          if mo != co:
            n_disagreements += 1
          if mo != 1:
            mo_errors += 1
          if co != 1:
            co_errors += 1
      print('\n')
      print('----------------')
      print('Disagreement: ', n_disagreements/n_instances)
      print('TensorFlow Accuracy: ', 1-mo_errors/n_instances)
      print('Compiler Accuracy: ', 1-co_errors/n_instances)
      print('\n')
      bar.finish()

  do = input('Test on random-valued mfccs? y/n\n')
  if do == 'y':
    n_instances = 1000
    mo_errors = 0
    co_errors = 0
    n_disagreements = 0
    inputs = np.random.uniform(low=-100, high=100, size=(n_instances, 
    	                                                       args.n_features))
    for x in inputs:
      if SCALE:
        x = myscale(x)
      co = 1 #if classify2(x) else 0
      if args.tensorflow:
        mo = tfclassify(np.reshape(x, (1, args.n_features)))
      else:
        mo = tfclassify(x)
      if mo != co:
        n_disagreements += 1
      if mo != 0:
        mo_errors += 1
      if co != 0:
        co_errors += 1
    print('\n')
    print('----------------')
    print('Disagreement: ', n_disagreements/n_instances)
    print('TensorFlow Accuracy: ', 1-mo_errors/n_instances)
    print('Compiler Accuracy: ', 1-co_errors/n_instances)
    print('\n')

  do = input('Test on specific wavs? y/n\n')
  if do == 'y':

    paths = ('../nonspeech.wav', '../speech.wav', 
      '../nonspeech2.wav', '../speech2.wav')

    for path in paths:
      n_disagreements = 0
      co_errors = 0
      mo_errors = 0

      audio, sr = librosa.load(path, sr=16000)
      mfccs = XTRACT(audio, sr, alpha=ALPHA, beta=BETA, 
      	n_interleavings=N_INTERLEAVINGS)
      n_instances = mfccs.shape[0]

      if SCALE:
          mfccs = myscale(mfccs)

      if 'nonspeech' in path:
        gt = 0
      else:
        gt = 1

      for i in range(mfccs.shape[0]):
        co = 1 #if classify2(mfccs[i]) else 0
        if args.tensorflow:
          mo = tfclassify(np.reshape(mfccs[i], (1, args.n_features)))
        else:
          mo = tfclassify(mfccs[i])
        if co != gt:
          co_errors += 1
        if mo != gt:
          mo_errors += 1
        if co != mo:
          n_disagreements += 1
      
      print('\n')
      print(path)  
      print('----------------')
      print('Disagreement: ', n_disagreements/n_instances)
      print('TensorFlow Accuracy: ', 1-mo_errors/n_instances)
      print('Compiler Accuracy: ', 1-co_errors/n_instances)
      print('\n')

  do = input('Test on training data? y/n\n')
  if do == 'y':
    
    n_disagreements = 0
    co_errors = 0
    mo_errors = 0

    n_instances = 1000

    train = load_csv(case='train', reuse=True, data_path=args.path)
    X, Y = train.next_batch(n_instances)
    if SCALE:
      X = myscale(X)
    for x, y in zip(X,Y):
      gt = int(np.argmax(y))
      co = 1 #if classify2(x) else 0
      if args.tensorflow:
        mo = tfclassify(np.reshape(x, (1, args.n_features)))
      else:
        mo = tfclassify(x)
      if co != mo:
        n_disagreements += 1
      if mo != gt:
        mo_errors += 1
      if co != gt:
        co_errors += 1
    print('\n')
    print('----------------')
    print('Disagreement: ', n_disagreements/n_instances)
    print('TensorFlow Accuracy: ', 1-mo_errors/n_instances)
    print('Compiler Accuracy: ', 1-co_errors/n_instances)
    print('\n')

  do = input('Test on your voice? y/n\n')
  if do == 'y':
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 16000
    CHUNK = int(RATE*0.05)

    p = pyaudio.PyAudio()
    
    stream = p.open(format=FORMAT,
                  channels=CHANNELS,
                  rate=RATE,
                  input=True,
                  frames_per_buffer=CHUNK)
    
    i = 0
    time_elapsed = 0
    last_200ms = [None for _ in range(4)]
    while True:
      data = stream.read(CHUNK)
      data = np.frombuffer(data, dtype=np.float32)
      
      xf = np.squeeze(XTRACT(data, RATE, alpha=ALPHA, beta=BETA, 
      	n_interleavings=N_INTERLEAVINGS))
      if SCALE:
      	xf = myscale(xf)
      if args.tensorflow:
        xf = np.reshape(xf, (1, args.n_features))
      last_200ms[i] = 1 if tfclassify(xf) else 0

      i = (i + 1) % 4
      if i % 4 == 0:
        print(sum(last_200ms) >= 2)
      time_elapsed += 0.05
      if time_elapsed >= 10:
        break

    stream.stop_stream()
    stream.close()
    p.terminate()

