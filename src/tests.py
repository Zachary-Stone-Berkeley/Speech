import argparse
import numpy as np
import string
# ------------
import amitools
import search_utils


ALPHABET = string.ascii_uppercase + string.ascii_lowercase + "".join(
  [str(n) for n in range(10)]) + ',.?!'
ALPHABET = [char for char in ALPHABET]
N_CHARS = len(ALPHABET)

def rand_float():
  return np.random.uniform(low=0.0, high=100.0)

def sample_string(alphabetic=False):
  if alphabetic:
    word = ''
    while True:
      word = word + ALPHABET[np.random.randint(low=0, high=51)]
      if np.random.randint(low=0, high=9) > 4:
        break
  else:
    word = ALPHABET[52+np.random.randint(low=0, high=13)]
  return word

def make_synthetic_xml(n_segments=100):
  
  def return_seg_type(i, type_of_first_seg):
    if i % 2 == 0:
      return type_of_first_seg
    else:
      if type_of_first_seg == 'speech':
        return 'non-speech'
      else:
        return 'speech'
  
  row = '<w nite:id="TEST.words{}" starttime="{}" endtime="{}">{}</w>\n'

  with open('../datadir/synthetic_xml.xml', 'w+') as file:
    breaks = [0.]
    for i in range(1, n_segments):
      breaks.append(breaks[i-1] + rand_float())

    type_of_first_seg = 'speech' if rand_float() > 50 else 'non-speech'
    speech_segments, non_speech_segments = [], []

    for i in range(n_segments-1):
      seg_type = return_seg_type(i, type_of_first_seg)
      seg_start, seg_end = breaks[i], breaks[i+1]

      if seg_type == 'speech':
        speech_segments.append((seg_start, seg_end))
      else:
        non_speech_segments.append((seg_start, seg_end))


      utterance_starts = [0.]
      while utterance_starts[-1] + seg_start < seg_end:
        utterance_starts.append(utterance_starts[-1] + np.random.uniform(
          low=0., high=2) + 0.01)
      utterance_starts[-1] = seg_end-seg_start
      for j in range(len(utterance_starts)-1):
        start = utterance_starts[j]+seg_start 
        end = utterance_starts[j+1]+seg_start
        if seg_type == 'speech':
          word = sample_string(True)
          file.write(row.format(str(i), str(start), str(end), word))
        elif rand_float() > 50:
          word = sample_string(False)
          file.write(row.format(str(i), str(start), str(end), word))


  return speech_segments, non_speech_segments, breaks[-1]

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--test_cases", nargs='+', default='')
  args = parser.parse_args()
  
  if '1' in args.test_cases:
    for _ in range(1000):
      speech_segments, non_speech_segments, duration = \
        make_synthetic_xml(100)
      if len(speech_segments) > 1 and len(non_speech_segments) > 1:
        tags_to_words = {'TEST':['../datadir/synthetic_xml.xml']}
        tags_to_lengths = {'TEST':duration}
        tags2speech, tags2nonspeech = amitools.get_speech_segments(['TEST'], 
          tags_to_words, tags_to_lengths)

        for i, (start, end) in enumerate(speech_segments):
          assert tags2speech['TEST'][i] == (start, end)
        for i, (start, end) in enumerate(non_speech_segments):
          assert tags2nonspeech['TEST'][i] == (start, end)
        for i, (start, end) in enumerate(tags2speech['TEST']):
          assert speech_segments[i] == (start, end)
        for i, (start, end) in enumerate(tags2nonspeech['TEST']):
          assert non_speech_segments[i] == (start, end)
    print("Test 1 passed, get_speech_segments() is working.")
  
  if '2' in args.test_cases:
    size = 100
    for _ in range(10):
      n_incorrect, n_valid = 0, 0
      hop_length_in_s, window_length_in_s = 0.01, 0.03
      speech_segments, non_speech_segments, duration = \
        make_synthetic_xml(size)

      tags_to_words = {'TEST':['../datadir/synthetic_xml.xml']}
      tags_to_lengths = {'TEST':duration}
      tags2speech, _ = amitools.get_speech_segments(['TEST'], tags_to_words, \
        tags_to_lengths)
            
      labels = amitools.make_labels('TEST', hop_length_in_s, window_length_in_s,
        irange=int(duration//hop_length_in_s), search_type='start',
        tags_to_words=tags_to_words, tags_to_lengths=tags_to_lengths)

      start, end = 0., 0. + hop_length_in_s
      for i in range(len(labels)):
        ground_truth = search_utils.slow_search(tags2speech['TEST'], start, \
          window_length_in_s)
        if ground_truth != -1:
          n_valid += 1
          if labels[i] != ground_truth:
            n_incorrect += 1
        start += hop_length_in_s
        end += hop_length_in_s
        start = np.round(start, 2)
        end = np.round(end, 2)
      error_rate = n_incorrect/n_valid
      assert error_rate < 0.00001
    print(f'Test 2 passed, make_labels() is working.')

  if '3' in args.test_cases:

    size = 1000

    for _ in range(1000):

      a = np.zeros(shape=(size, 2))
      a[0][1] = rand_float()
      for i in range(1, size):
        a[i][0] = a[i-1][1] + rand_float()
        a[i][1] = a[i][0] + rand_float()
      a = a.tolist()

      minimum, maximum = a[0][0], a[-1][-1]
      for _ in range(1000):
        val = np.random.uniform(low=minimum, high=maximum)
        ground_truth = search_utils.slow_search(a, val, 0.03)
        res = search_utils.binary_search_by_start(a, 0, len(a)-1, val, 0.03)
        try:
          assert ground_truth == res
        except Exception as e:
          print('\n')
          print(f'Ground truth is {ground_truth} but result is {res} when \
            labeling the interval [{val}, {val+0.03}].\n')
          raise e

    print('Test 3 passed, binary_search_by_start() is working.')