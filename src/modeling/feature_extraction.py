import numpy as np
import librosa

def xtract(audio, sr, alpha=0.01, beta=0.01, n_interleavings=5,
  start_index=1, n_mfcc=20):

  '''
  audio: raw audio
  sr: sampling rate
  alpha: window/sub-window size
  beta: hop length
  n_interleavings: number of sub windows
  '''

  frames_per_window, hop_length = int(sr*alpha), int(sr*beta)
  window_length_in_s = alpha + (n_interleavings-1)*beta

  mfccs = librosa.feature.mfcc(audio, sr=sr, 
    n_mfcc=n_mfcc, n_fft=frames_per_window, 
    hop_length=hop_length, center=False,
    n_mels=128)
  
  n_instances = mfccs.shape[1]-n_interleavings+1
  n_xfeatures = (n_mfcc-start_index)*n_interleavings

  res = np.zeros((n_instances, n_xfeatures))
  for i in range(n_instances):
    res[i] = mfccs[start_index:, i:i+n_interleavings].reshape(n_xfeatures)
  return res



