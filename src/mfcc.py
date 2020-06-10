import os
#os.environ['W'] = 'ignore'
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

total = 72984
smallest_class = 972
DCASE_probs = {}
DCASE_probs['absence'] = smallest_class/18860
DCASE_probs['cooking'] = smallest_class/5124
DCASE_probs['dishwashing'] = smallest_class/1424
DCASE_probs['eating'] = smallest_class/2308
DCASE_probs['other'] = smallest_class/2060
DCASE_probs['social_activity'] = smallest_class/4944
DCASE_probs['vacuum_cleaner'] = smallest_class/972
DCASE_probs['watching_tv'] = smallest_class/18648
DCASE_probs['working'] = smallest_class/18644

SPEECH_LOCATION = '../datadir/speech_sources/'
NONSPEECH_LOCATION = '../datadir/nonspeech_sources/'
MUSIC_LOCATION = '../datadir/music_sources/'
CSV_NAME = '{}.csv'

def get_paths(source):
  if isinstance(source, str):
    return get_paths_single(source)
  elif isinstance(source, list):
    return get_paths_multisource(source)
  else:
    raise ValueError

def get_paths_single(source):
  root = f'../{source}/'
  paths = [[x[0] + '/' + y for y in x[2] if 
    any(extension in y for extension in EXTENSIONS)] for 
      x in os.walk(root)]
  paths = reduce((lambda a,b: a+b), paths)
  np.random.shuffle(paths)
  return paths

def get_paths_multisource(sources):
  res = []
  for source in sources:
    root = f'../{source}/'
    paths = [[x[0] + '/' + y for y in x[2] if 
      any(extension in y for extension in EXTENSIONS)] for 
        x in os.walk(root)]
    paths = reduce((lambda a,b: a+b), paths)
    np.random.shuffle(paths)
    res.extend(paths)
  np.random.shuffle(res)
  return res

def make_mfcc_data(sources, alpha=0.03, beta=0.01, n_interleavings=1, n_mfcc=19, 
                   start_index=1, center=False, max_gb=25., coarse=False,
                   audio_sr=None, randomize_sr=False, subdir='', n_mels=128,
                   fmax=None):

  if subdir:
    subdir = subdir + '/'
    for parent_dir in ['speech_sources', 'nonspeech_sources', 'music_sources']:
      savepath = f'../datadir/{parent_dir}/{subdir}'
      if not os.path.exists(savepath):
        os.makedirs(savepath)
  
  n_errors = 0
  n_cols = (n_mfcc-start_index)*n_interleavings + 1
  max_rows = int(max_gb*BITS_PER_GB/(64*n_cols))
  
  nonspeech_source, speech_source, music_source = sources  
  speech_paths, nonspeech_paths, music_paths = [], [], []

  if speech_source:
    speech_paths = get_paths(speech_source)

  if music_source:
    music_paths = get_paths(music_source)

  if nonspeech_source:
    nonspeech_paths = []
    with open('../DCASE/meta.txt', 'r') as f:
      for line in f:
        try:
          path_to_wav, activity, idx = line.strip().split('\t')
          if activity in ['social_activity', 'watching_tv']:
            continue
          if activity not in DCASE_probs:
            print(activity)
            raise Exception
          if np.random.uniform() > 2*DCASE_probs[activity]:
            continue
          nonspeech_paths.append('../DCASE/' + path_to_wav)
        except:
          continue
    np.random.shuffle(nonspeech_paths)

  with open(SPEECH_LOCATION + subdir + CSV_NAME.format(
         'train'), 'w+') as speech_train_file, \
       open(SPEECH_LOCATION + subdir + CSV_NAME.format(
         'test'), 'w+') as speech_test_file, \
       open(NONSPEECH_LOCATION + subdir + CSV_NAME.format(
         'train'), 'w+') as nonspeech_train_file, \
       open(NONSPEECH_LOCATION + subdir + CSV_NAME.format(
         'test'), 'w+') as nonspeech_test_file, \
       open(MUSIC_LOCATION + subdir + CSV_NAME.format(
         'train'), 'w+') as music_train_file, \
       open(MUSIC_LOCATION + subdir + CSV_NAME.format(
         'test'), 'w+') as music_test_file:
    
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
          break

        is_ami = 'amicorpus' in path
        is_rhet = 'american_rhetoric' in path
        is_dcase = 'DCASE' in path
        is_fma = 'fma_medium' in path
        is_mnet = 'musicnet' in path

        if is_rhet and 'top100' not in path and 'amicorpus' in speech_source:
          if np.random.uniform() < 0.75:
            continue
        
        if is_ami:
          tag = re.search("([a-zA-Z]{2}[0-9]{4}[a-zA-Z]*)", path).group(0)
          part = get_ami_part(tag)
        elif is_dcase or is_fma or is_rhet:
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
        
        try:
          if randomize_sr:
            sr = np.random.randint(low=16000, high=44100)
          else:
            sr = audio_sr
          audio, sr = librosa.load(path, sr=sr)
        except:
          n_errors += 1
          if n_errors > 1000:
            print('Too many errors. Halting.')
            raise Exception
          continue

        if sr == 0: # small number of the fma_medium audio samples have sr = 0
          continue

        frames_per_window, hop_length = int(sr*alpha), int(sr*beta)
        window_length_in_s = alpha + (n_interleavings-1)*beta
        
        if len(audio) < frames_per_window:
        	continue

        mfccs = librosa.feature.mfcc(audio, sr=sr, 
          n_mfcc=n_mfcc, n_fft=frames_per_window, 
          hop_length=hop_length, center=center,
          n_mels=n_mels, fmax=fmax)
        
        if coarse:
          mfccs2 = librosa.feature.mfcc(audio, sr=sr,
            n_mfcc=n_mfcc, n_fft=int(sr*window_length_in_s),
            hop_length=hop_length, center=center,
            n_mels=n_mels, fmax=fmax)

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
  parser.add_argument('-a', 
                      '--alpha', 
                      type=float, 
                      default=0.01, 
  	                  help="Window length of the MFCC in seconds.")

  parser.add_argument('-b', 
                      '--beta', 
                      type=float, 
                      default=0.01, 
                      help="Hop length of the mfcc in seconds.")
  
  parser.add_argument('-i', 
                      '--n_interleavings', 
                      type=int, 
                      default=5, 
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
  
  parser.add_argument('--coarse', action='store_true')
  
  parser.add_argument('--n_mels', type=int, default=128)
  
  parser.add_argument('--fmax', type=int, default=None)

  # arguments relating to the sourced data
  parser.add_argument('--sources', '-s', type=str, nargs='+', action='append',
    default=[])
  parser.add_argument('--sampling_rate', '-sr', type=int, default=16000)
  parser.add_argument('--randomize_sr', '-r', type=str2bool, default=False)
  parser.add_argument('--subdir', type=str, default='')
  
  # other
  parser.add_argument('-gb', '--max_gb', type=float, default=1.,
    help="Maximum size of an array in GBs containing the speech (or nonspeech \
    data. Usually than the array in .csv format.")
  
  args = parser.parse_args()
  if not args.sources:
    args.sources = ['DCASE', ['amicorpus', 'american_rhetoric'], 'musicnet']
  args.sources = [x if any(x) else '' for x in args.sources]
  print('Sources = ', args.sources)
  print('Coarse = ', args.coarse)

  make_mfcc_data(sources=args.sources,
  	             alpha=args.alpha, 
                 beta=args.beta, 
                 n_interleavings=args.n_interleavings, 
                 n_mfcc=args.n_mfcc, 
                 start_index=args.start_index, 
                 center=args.center,
                 max_gb=args.max_gb,
                 coarse=args.coarse,
                 audio_sr=args.sampling_rate,
                 randomize_sr=args.randomize_sr,
                 subdir=args.subdir,
                 n_mels=args.n_mels,
                 fmax=args.fmax)


# python mfcc.py -s 'DCASE' -s 'amicorpus' 'american_rhetoric' -s 'fma_medium' --subdir 'rhet' --gb 10


