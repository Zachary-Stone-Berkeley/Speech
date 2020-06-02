import numpy as np
import os

class DataSet(object):

  def __init__(self, list_of_data):
    
    self.list_of_data = list_of_data
    self.epochs_completed = 0
    self.index_in_epoch = 0
    self.num_examples = list_of_data[0].shape[0]
    self.n_features = list_of_data[0].shape[1]
    self.num_data = len(self.list_of_data)
  
  def shuffle(self):
    perm = np.arange(self.num_examples)
    np.random.shuffle(perm)
    for i in range(self.num_data):
      self.list_of_data[i] = self.list_of_data[i][perm]

  def next_batch(self, batch_size, shuffle=True):

    start = self.index_in_epoch
    
    if self.epochs_completed == 0 and start == 0 and shuffle:
      self.shuffle()

    if start + batch_size <= self.num_examples:
      self.index_in_epoch += batch_size
      end = self.index_in_epoch
      return [self.list_of_data[i][start:end] for i in range(self.num_data)]

    if start + batch_size > self.num_examples:
      self.epochs_completed += 1
      rest_num_examples = self.num_examples - start
      rest_part = [self.list_of_data[i][start:self.num_examples] for i 
        in range(self.num_data)]
      if shuffle:
        self.shuffle()
      start = 0      
      self.index_in_epoch = batch_size - rest_num_examples
      end = self.index_in_epoch
      new_part = [self.list_of_data[i][start:end] for i in range(self.num_data)]      
      return [np.concatenate((rest_part[i], new_part[i]), axis=0) for i 
        in range(self.num_data)]

class DataSetMemFriendly(object):
  
  # In progress
  # Useful if data can't be loaded into memory

  def __init__(self, speech_src, nonspeech_src, size, rows_per_chunk, 
    n_classes=2):
    
    with open(speech_src, 'r') as f:
      self.n_features = len(f.readline().split(',')[:-1])
    
    # source information
    self.sources = (nonspeech_src, speech_src)
    self.csvs = [open(src, 'r') for src in self.sources]
    self.n_classes = n_classes
    self.n_sources = len(self.sources)
    size = size + (size % size//self.n_sources)
    assert size/self.n_sources == size//self.n_sources
    self.num_examples = size
    
    # chunk book keeping
    self.rows_per_chunk = rows_per_chunk
    self.load_new_chunk = True
    self.chunk_i = 0
    self.chunk_obs = np.zeros((rows_per_chunk, self.n_features))
    self.chunk_labels = np.zeros((rows_per_chunk, 2)).astype(np.int32)

    self.batch_cuts = np.arange(self.rows_per_chunk, step=64)

  def get_line(self, csv):
    return csv.readline().strip().split(',')

  def load_chunk(self):
    size = self.rows_per_chunk//self.n_sources

    for isrc, source in enumerate(self.sources):

      start = int(isrc*size)
      end = int((isrc+1)*size)
      for i in range(start, end):
        line = self.get_line(self.csvs[isrc])
        if len(line) != 96:
          if line == ['']:
            self.csvs[isrc].close()
            self.csvs[isrc] = open(self.sources[isrc], 'r')
          continue
        label = int(line.pop())
        self.chunk_obs[i] = line
        self.chunk_labels[i] = [1 if i == label else 0 for i in 
          range(self.n_sources)]
   
    perm = np.arange(self.rows_per_chunk)
    np.random.shuffle(perm)
    self.chunk_obs = self.chunk_obs[perm]
    self.chunk_labels = self.chunk_labels[perm]

  def next_batch(self, batch_size, shuffle=True):

    if self.load_new_chunk:
      self.load_chunk()
      self.batch_i = 0
      self.load_new_chunk = False
    
    start = self.batch_i
    self.batch_i += batch_size
    end = self.batch_i
    
    if end > self.rows_per_chunk:
      self.load_new_chunk = True
      end = self.rows_per_chunk

    return [self.chunk_obs[start:end], self.chunk_labels[start:end]]



