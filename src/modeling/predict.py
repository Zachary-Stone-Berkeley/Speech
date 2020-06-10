import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
sys.path.append('../../../Daimensions/')
import tensorflow as tf
import argparse
import numpy as np
import librosa
import pyaudio
import matplotlib.pyplot as plt
import soundfile as sf
from functools import reduce
from progress.bar import Bar
from sklearn.preprocessing import scale
# ------
from out import classify, single_classify, transform
#from detector import classify, single_classify, transform
#from sixteen import classify, single_classify
#from fourtyfour import classify, single_classify
from DataGetter import load_csv
from models import AudioClassifier, ComplexAudioClassifier
from feature_extraction import xtract, xtract_coarse
from tf_model import tf_classify
#from scaling import mu, var

#def transform(x):
#  return x

SCALE = False
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
  parser.add_argument('-nn', '--n_neurons', type=int, default=10, 
    help="number of neurons")
  parser.add_argument('-nf', '--n_features', type=int, default=95,
    help="number of input features")
  parser.add_argument('-nc', '--n_classes', type=int, default=3)
  parser.add_argument('-oh', '--onehot', type=str2bool, default=True)
  parser.add_argument('-tf', '--tensorflow', type=str2bool, default=True)
  parser.add_argument('--conv', action='store_true')
  parser.add_argument('-p', '--pool', type=str2bool, default=False)
  parser.add_argument('--path', type=str, default='')
  parser.add_argument('--name', type=str, default='')
  parser.add_argument('-d', '--do', type=str, default='nnnny')
  parser.add_argument('--audio_sr', '-asr', type=int, default=16000)
  parser.add_argument('--mfcc_sr', '-msr', type=int, default=None)
  parser.add_argument('--plot', type=str2bool, default=False)
  args = parser.parse_args()

  audio_sr = args.audio_sr
  mfcc_sr = args.mfcc_sr

  if args.audio_sr == -1:
    args.audio_sr = None

  if not args.name:
    if args.conv:
      name = 'conv'
    else:
      name = 'feedforward'
  else:
    name = args.name

  if 'w' in args.path:
    XTRACT = xtract
    ALPHA = 0.2
    BETA = 0.2
    N_INTERLEAVINGS = 10
  elif 'coarse' in args.path or 'test' in args.path:
    XTRACT = xtract_coarse
    ALPHA = 0.01
    BETA = 0.01
    N_INTERLEAVINGS = 5
  else:
    XTRACT = xtract
    ALPHA = 0.01
    BETA = 0.01
    N_INTERLEAVINGS = 5

  if 'h' in args.path:
    N_MELS = 256
    FMAX = None
  else:
    N_MELS = 128
    FMAX = None

  print(ALPHA, BETA, N_INTERLEAVINGS, N_MELS, FMAX)

  if args.tensorflow:
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    if not args.conv:
      c = AudioClassifier(n_neurons=args.n_neurons, sess=sess, 
        n_features=args.n_features, n_classes=args.n_classes, name=name,
        onehot_labels=args.onehot, pool=args.pool)
    else:
      c = ComplexAudioClassifier(n_neurons=args.n_neurons, sess=sess, 
        n_features=args.n_features, n_classes=args.n_classes, name=name,
        onehot_labels=args.onehot)
    if args.path:
      c.load_weights(f'./pretrained models/{args.path}/{c.name}.ckpt')
    else:
      c.load_weights(f'./pretrained models/{c.name}.ckpt')

  
  if False:
    root = '../../'
    fma = root + 'fma_medium/000/000002.mp3'
    data, samplerate = sf.read(fma)
    print(data.__attrs__, samplerate)
    raise Exception

  print('\n\n[other datasets][random inputs][training data][voice][selected audio]\n')

  if args.do[0] == 'y':
    size = 100
    for dataset in ('LibriSpeech', 'Vox Forge', 'Flickr Audio Caption', 'common-voice', 'FSD Kaggle'):
      paths = get_paths(dataset)
      np.random.shuffle(paths)
      bar = Bar(dataset, max=size)
      n_disagreements = 0
      co_errors = 0
      mo_errors = 0
      n_instances = 0
      if dataset == 'FSD Kaggle':
        label = 0
      else:
        label = 1
      
      for wav in paths[:size]:
        bar.next()
        sr = 16000
        audio, sr = librosa.load(wav, sr=audio_sr)
        if len(audio) < 0.5:
          continue
        xf = XTRACT(audio, mfcc_sr, alpha=ALPHA, beta=BETA, 
        	n_interleavings=N_INTERLEAVINGS, n_mels=N_MELS, fmax=FMAX)
        if SCALE:
          xf = myscale(xf)
        n_instances += xf.shape[0]
        co = np.asarray(classify(transform(xf)))
        mo = c.predict(xf)
        mo_errors += mo.shape[0] - np.sum(mo == label)
        co_errors += co.shape[0] - np.sum(co == label)
        n_disagreements += np.sum(1-np.equal(co, mo))
      print('\n')
      print('----------------')
      print('Disagreement: ', n_disagreements/n_instances)
      print('TensorFlow Accuracy: ', 1-mo_errors/n_instances)
      print('Compiler Accuracy: ', 1-co_errors/n_instances)
      print('\n')
      bar.finish()

  if args.do[1] == 'y':
    n_instances = 1000
    mo_errors = 0
    co_errors = 0
    n_disagreements = 0
    inputs = np.random.uniform(low=-100, high=100, size=(n_instances, 
    	                                                       args.n_features))
    for x in inputs:
      co = single_classify(transform(x))
      if SCALE:
        x = myscale(x)
      mo = c.predict(np.reshape(x, (1, args.n_features)))
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

  if args.do[2] == 'y':
    
    n_disagreements = 0
    co_errors = 0
    mo_errors = 0

    train = load_csv(case='train', reuse=True, dest=args.path)
    datasize = train.num_examples//10
    batch_size = 64
    n_batches = datasize//batch_size

    for i in range(n_batches):
      #print(f'{i} of {n_batches}')
      X, Y = train.next_batch(batch_size)
      Y = np.argmax(Y, axis=1)
      co = classify(transform(X))
      if SCALE:
        X = myscale(X)
      mo = c.predict(X)
      co_errors += np.sum(1-np.equal(co, Y))
      mo_errors += np.sum(1-np.equal(mo, Y))
      n_disagreements += np.sum(1-np.equal(co, mo))

    print('\n')
    print('----------------')
    print('Disagreement: ', n_disagreements/(batch_size*n_batches))
    print('TensorFlow Accuracy: ', 1-mo_errors/(batch_size*n_batches))
    print('Compiler Accuracy: ', 1-co_errors/(batch_size*n_batches))

  if args.do[3] == 'y':
    ctype = input('Use compiler (c) or tensorflow (t).')
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = args.audio_sr
    if mfcc_sr == None:
      mfcc_sr = RATE
    CHUNK = int(RATE*0.05)

    p = pyaudio.PyAudio()
    
    stream = p.open(format=FORMAT,
                  channels=CHANNELS,
                  rate=RATE,
                  input=True,
                  frames_per_buffer=CHUNK)
    
    time_elapsed = 0
    while True:
      data = stream.read(CHUNK)
      data = np.frombuffer(data, dtype=np.float32)
      
      xf = np.squeeze(XTRACT(data, mfcc_sr, alpha=ALPHA, beta=BETA, 
      	n_interleavings=N_INTERLEAVINGS, n_mels=N_MELS, fmax=FMAX))
      if SCALE:
      	xf = myscale(xf)
      if args.tensorflow:
        xf = np.reshape(xf, (1, args.n_features))

      if ctype == 'c':
        pred = classify(transform(xf))[0]
      elif ctype == 't':
        pred = c.predict(xf)[0]
      else:
        raise ValueError

      if pred == 0:
        print('nonspeech')
      elif pred == 1:
        print('speech')
      elif pred == 2:
        print('music')
      else:
        raise Exception

      time_elapsed += 0.05
      if time_elapsed >= 10:
        break

    stream.stop_stream()
    stream.close()
    p.terminate()
  
  if args.do[4] == 'y':
    
    paths = ['/home/zachary/Documents/AMI/fma_medium/000/000002.mp3',
             '/home/zachary/Downloads/Mimicking Birds - Bloodlines.mp3',
             '/home/zachary/Downloads/988-v07.lehman1.ogg',
             #'/home/zachary/Documents/AMI/DCASE/audio/DevNode1_ex1_1.wav',
             '/home/zachary/Downloads/8kHz.wav',
             '/home/zachary/Downloads/16kHz.wav',
             '/home/zachary/Downloads/44.1kHz.wav',
             '/home/zachary/Downloads/dream.mp3',
             '/home/zachary/Downloads/dream.wav',
             '/home/zachary/Downloads/perils.mp3',
             '/home/zachary/Downloads/holocaust.mp3',
             '/home/zachary/Downloads/shaw2.mp3',
             '/home/zachary/Downloads/copy.mp3',
             '/home/zachary/Downloads/myvoice.mp3',
             '/home/zachary/Downloads/myvoice2.wav',
             '/home/zachary/Downloads/myvoice3.wav']
  
    if 0:
      paths = paths + ['/home/zachary/Documents/AMI/amicorpus/EN2001a/audio/EN2001a.Mix-Headset.wav',
                       '/home/zachary/Downloads/barackobamabatonrougefloodingARXE.mp3',
                       '/home/zachary/Downloads/trumpspeech.mp3']
    
    speech = []
    nonspeech = []
    music = []

    '''
    test with 16000 for audio_sr but original sr of audio for mfcc_sr
    so fma_medium is audio_sr=16000 and mfcc_sr=44100
    '''
    if args.plot:
      paths = paths[-5:-3]
      srs = list(range(16000, 44100, 400))
    else:
      paths = paths
      srs = [args.audio_sr]

    for path in paths:
      
      for audio_sr in srs:
        audio, sr = librosa.load(path, sr=audio_sr)
        print('\n\n\n', path, sr)

        if mfcc_sr:
          xf = XTRACT(audio, mfcc_sr, alpha=ALPHA, beta=BETA, 
                n_interleavings=N_INTERLEAVINGS, n_mels=N_MELS, fmax=FMAX)
        else:
          xf = XTRACT(audio, sr, alpha=ALPHA, beta=BETA, 
                n_interleavings=N_INTERLEAVINGS, n_mels=N_MELS, fmax=FMAX)
        co = 1#np.asarray(classify(transform(xf)))
        mo = c.predict(xf)

        if args.plot:
          nonspeech.append(np.sum(mo == 0) / mo.shape[0])
          speech.append(np.sum(mo == 1) / mo.shape[0])
          music.append(np.sum(mo == 2) / mo.shape[0])
          continue

        
        for i in range(3):
          if i == 0:
            pred = 'nonspeech'
          elif i == 1:
            pred = 'speech'
          elif i == 2:
            pred = 'music'
          print(f'TensorFlow predicts {np.sum(mo == i) / mo.shape[0]} of the sample is {pred}')
          #print(f'Brainome predicts   {np.sum(co == i) / co.shape[0]} of the sample is {pred}')
        
    if args.plot:
      plt.plot(srs, speech, label='speech')
      plt.plot(srs, nonspeech, label='nonspeech')
      plt.plot(srs, music, label='music')
      plt.legend()
      plt.show()


'''
python -W ignore mfcc.py -gb 1.2 --subdir '8000'
python main.py --path 'None'
python -W ignore predict.py --path 'None' -d nnnny

python -W ignore predict.py --path '16000c' -asr 16000 -nn 10
python -W ignore predict.py --path '16000' -asr 16000
python -W ignore predict.py --path 'ww' -asr 16000

btc -v -v -f NN -o sixteen.py -server qa3-daimensions.brainome.ai './datadir/16khz.csv'
btc -v -v -f NN -o new.py -server qa3-daimensions.brainome.ai new.csv

ffmpeg -i shaw.mp3 -ss 00:00:10 -to 00:00:40 -c copy shaw.wav
'''