import os
import librosa
import argparse
import re
import numpy as np
from progress.bar import Bar
from functools import reduce
# ------------
from amitools import make_labels, get_ami_part

BITS_PER_GB = 8589934592
EXTENSIONS = ['.wav', '.mp3', '.flac']

SPEECH_LOCATION = '../datadir/speech_sources/{}_{}.csv'
if not os.path.exists('../datadir/speech_sources/'):
  os.makedirs('../datadir/speech_sources/')

NONSPEECH_LOCATION = '../datadir/nonspeech_sources/{}_{}.csv'
if not os.path.exists('../datadir/nonspeech_sources/'):
  os.makedirs('../datadir/nonspeech_sources/')

MUSIC_LOCATION = '../datadir/music_sources/{}_{}.csv'
if not os.path.exists('../datadir/music_sources/'):
  os.makedirs('../datadir/music_sources/')

def get_paths(source):
  root = f'../{source}/'
  paths = [[x[0] + '/' + y for y in x[2] if 
    any(extension in y for extension in EXTENSIONS)] for 
      x in os.walk(root)]
  paths = reduce((lambda a,b: a+b), paths)
  np.random.shuffle(paths)
  return paths

def make_mfcc_data(sources, alpha=0.03, beta=0.01, n_interleavings=1, n_mfcc=19, 
                   start_index=1, center=False, max_gb=25., coarse=False):

  n_cols = (n_mfcc-start_index)*n_interleavings + 1
  max_rows = int(max_gb*BITS_PER_GB/(64*n_cols))
  
  speech_source, nonspeech_source, music_source = sources
  
  speech_paths, nonspeech_paths = [], []
  if speech_source:
    speech_paths = get_paths(speech_source)
  if nonspeech_source:
    nonspeech_paths = get_paths(nonspeech_source)
  if music_source:
    music_paths = get_paths(music_source)

  with open(SPEECH_LOCATION.format(speech_source, 'train'), 'w+') \
         as speech_train_file, \
       open(SPEECH_LOCATION.format(speech_source, 'test'), 'w+') \
         as speech_test_file, \
       open(NONSPEECH_LOCATION.format(nonspeech_source, 'train'), 'w+') \
         as nonspeech_train_file, \
       open(NONSPEECH_LOCATION.format(nonspeech_source, 'test'), 'w+') \
         as nonspeech_test_file, \
       open(MUSIC_LOCATION.format(music_source, 'train'), 'w+') \
         as music_train_file, \
       open(MUSIC_LOCATION.format(music_source, 'test'), 'w+') \
         as music_test_file:
    
    source_map = {0:'non-speech', 1:'speech', 2:'music'}
    file_map = {0:{'train':nonspeech_train_file, 'test':nonspeech_test_file},
                1:{'train':speech_train_file, 'test':speech_test_file},
                2:{'train':music_train_file, 'test':music_test_file}}
    
    for pid, paths in enumerate((nonspeech_paths, speech_paths, music_paths)):

      n_rows = 0
      bar = Bar("Making MFCCs for {} data.".format(source_map[pid]), 
        max=len(paths))

      for path in paths:

        bar.next()

        if n_rows >= max_rows:
          print("\nMax rows reached.\n")
          break

        is_ami = \
            (nonspeech_source, speech_source, music_source)[pid] == 'amicorpus'
        is_dcase = \
            (nonspeech_source, speech_source, music_source)[pid] == 'DCASE'
        is_fma = \
            (nonspeech_source, speech_source, music_source)[pid] == 'fma_medium'
        
        if is_ami:
          tag = re.search("([a-zA-Z]{2}[0-9]{4}[a-zA-Z]*)", path).group(0)
          part = get_ami_part(tag)
        elif is_dcase or is_fma:
          part = 'train' if np.random.uniform() < 0.9 else 'test' # temp
        else:
          if 'train' in path:
            part = 'train'
          elif 'test' in path:
            part = 'test'
          else:
            part = 'val'
        if part == 'val':
            continue
        
        sr = None
        audio, sr = librosa.load(path, sr=sr)
        if sr == 0: # small number of the fma_medium audio samples have sr = 0
          continue
        frames_per_window, hop_length = int(sr*alpha), int(sr*beta)
        window_length_in_s = alpha + (n_interleavings-1)*beta
        if len(audio) < frames_per_window:
        	continue

        mfccs = librosa.feature.mfcc(audio, sr=sr, 
          n_mfcc=n_mfcc, n_fft=frames_per_window, 
          hop_length=hop_length, center=center,
          n_mels=128)
        
        if coarse:
          mfccs2 = librosa.feature.mfcc(audio, sr=sr,
            n_mfcc=n_mfcc, n_fft=int(sr*window_length_in_s),
            hop_length=hop_length, center=center,
            n_mels=128)

        if is_ami:
          labels = make_labels(tag, hop_length_in_s=beta, 
            window_length_in_s=window_length_in_s, irange=mfccs.shape[1])

        for i in range(mfccs.shape[1]):

          if is_ami:
            label = labels[i]
            if label != 1:
              continue
          else:
            label = pid
          
          if i <= mfccs.shape[1]-n_interleavings-1:
            if coarse:
              file_map[pid][part].write(",".join(str(x) for x in 
                mfccs[start_index:, i:i+n_interleavings].reshape(
                  (n_mfcc-start_index)*n_interleavings)) + ',' + \
                ",".join(str(x) for x in mfccs2[start_index:, i]) + \
                f',{label}\n')
            else:
              file_map[pid][part].write(",".join(str(x) for x in 
                mfccs[start_index:, i:i+n_interleavings].reshape(
                  (n_mfcc-start_index)*n_interleavings)) + f',{label}\n')
            n_rows += 1

      bar.finish()


if __name__ == "__main__":

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

  # arguments related to the extracted features
  parser.add_argument('-a', '--alpha', type=float, default=0.01, 
  	help="Window length of the MFCC in seconds.")
  parser.add_argument('-b', '--beta', type=float, default=0.01, 
  	help="Hop length of the mfcc in seconds.")
  parser.add_argument('-i', '--n_interleavings', type=int, default=5, 
  	help="Integer number of n_interleavings to use. If >1, each row of the \
    csv will be the MFCCs from n_interleavings \
  	consecutive windows.")
  parser.add_argument('--n_mfcc', type=int, default=20, 
  	help="The number of MFCCs to use.")
  parser.add_argument('--start_index', type=int, default=1, 
  	help="Integer index. Only MFCCs after start_index are written to the data \
    set.")
  parser.add_argument('--center', type=str2bool, default=False, 
  	help="Bool to center frames or not.")
  parser.add_argument('--coarse', type=str2bool, default=False)
  
  # arguments relating to the sourced data
  parser.add_argument('--sources', nargs='+', type=str,
    default=['amicorpus', 'DCASE', 'fma_medium'])
  #parser.add_argument('--speech_src', type=str, default='amicorpus',
  #	help="String to identify speech source.")
  #parser.add_argument('--nonspeech_src', type=str, default='DCASE',
  #	help="String to identify non-speech source.")
  #parser.add_argument('--music_src', type=str, default='fma_medium')
  
  # other
  parser.add_argument('--max_gb', type=float, default=15.,
    help="Maximum size of an array in GBs containing the speech (or nonspeech \
    data. Usually than the array in .csv format.")
  
  args = parser.parse_args()
  
  make_mfcc_data(sources=args.sources,
  	             alpha=args.alpha, 
                 beta=args.beta, 
                 n_interleavings=args.n_interleavings, 
                 n_mfcc=args.n_mfcc, 
                 start_index=args.start_index, 
                 center=args.center,
                 max_gb=args.max_gb,
                 coarse=args.coarse)





