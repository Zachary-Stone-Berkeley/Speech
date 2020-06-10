import numpy as np
import os
from sklearn.preprocessing import scale
# ---------------------------------
from CustomDataSet import DataSet, DataSetMemFriendly

BITS_PER_GB = 8589934592

def subsample(name=None, subdir='', max_gb=0.1):
  if name == None:
    print('name, subdir, max_gb')
    return None
  
  if subdir:
    subdir = subdir + '/'
  source = f'../../datadir/{subdir}combined_train.csv'
  size = os.path.getsize(source)/1000000000
  reject = max_gb/size

  with open(source, 'r') as file, \
       open(f'../../datadir/{name}.csv', 'w+') as small_file:

    for line in file:
      if np.random.uniform(low=0., high=1.) > reject:
        continue
      small_file.write(line)

def combine_sources(classmap, write_path, balance=False, 
  case='train', max_gb=1.0, data_path='', exact=False,
  include=None):

  if data_path:
    data_path = data_path + '/'

  n_rows = 0
  paths = []
  for clss in classmap:
    path = f'../../datadir/{clss}_sources/{data_path}{case}.csv'
    paths.append(path)
  
  print('Combining sources...')

  if balance:
    sizes = [os.path.getsize(paths[i]) for i in range(len(paths))]
    output_size = len(paths)*min(sizes)/1000000000
    base_reject = min(1., max_gb/output_size)
    if exact:
      nrows = []
      for path in paths:
        n = 0
        with open(path, 'r') as f:
          for line in f:
            if line:
              n += 1
        nrows.append(n)
      reject_probs = [min(1., min(nrows)/size) for size in nrows]
    else:
      reject_probs = [min(1., min(sizes)/size) for size in sizes]

  with open(write_path, 'w+') as c:
    for path_id, path in enumerate(paths):
      with open(path, 'r') as file:
        for line in file:
          reject = np.random.uniform(low=0., high=1.)
          if balance and reject > base_reject*reject_probs[path_id]:
            continue
          label = int(float(line.strip().split(',')[-1]))
          if label not in include:
            continue
          c.write(line)
          n_rows += 1

  return n_rows

def label_helper(line, n_classes, imap):
  label = int(float(line.strip().split(',')[-1]))
  return [0. if i != imap[label] else 1. for i in range(n_classes)]

def obs_helper(line):
  return line.strip().split(',')[:-1]

def load_csv(case='train', classmap=['nonspeech', 'speech', 'music'],
             max_gb=1.0, doscale=False, start=0, end=None, balance=True, 
             reuse=False, memory_friendly=False, data_path='', dest=''):
  
  if dest:
    dest = dest + '/'
  
  '''
  if memory_friendly:
    n_features = 95
    size = min(23618744, 6274324)
    max_rows = int(max_gb*BITS_PER_GB/(64*n_features))
    
    speech_path = '../../datadir/speech_sources/' + data_path 
    speech_path = speech_path + speech_src + '_train.csv'
    
    nonspeech_path = '../../datadir/nonspeech_sources/' + data_path 
    nonspeech_path = nonspeech_path + nonspeech_src + '_train.csv'
    
    data = DataSetMemFriendly(speech_src=speech_path, 
      nonspeech_src=nonspeech_path, size=size, rows_per_chunk=max_rows)
    return data
  '''

  imap = {}
  if 'nonspeech' in classmap:
    imap[0] = 0
    if 'speech' in classmap:
      imap[1] = 1
      if 'music' in classmap:
        imap[2] = 2
    elif 'music' in classmap:
      imap[2] = 1
  elif 'speech' in classmap:
    imap[1] = 0
    if 'music' in classmap:
      imap[2] = 1
  else:
    imap[2] = 0
  print(imap)
  print([x for x in imap])

  n_classes = len(classmap)
  path_to_csv = f'../../datadir/{dest}'

  if not os.path.exists(path_to_csv):
    os.makedirs(path_to_csv)

  csv_path = path_to_csv + f'combined_{case}.csv'
  if not reuse and os.path.exists(csv_path):
    warning = f'{csv_path} already exists. Would you like to overwrite? y/n\n'
    command = input(warning)
    if command == 'n':
      return
    elif command == 'y':
      print('\n')
      pass
    else:
      raise ValueError

  if not reuse or not os.path.exists(csv_path):
    n_rows = combine_sources(classmap, csv_path, include=[x for x in imap],
      balance=balance, case=case, max_gb=max_gb, data_path=data_path)

  n_rows = 1
  with open(csv_path.format(case), 'r') as file:
    n_features = len(file.readline().strip().split(',')) - 1
    for line in file:
      n_rows += 1

  obs = np.zeros(shape=(n_rows, n_features))
  labels = np.zeros(shape=(n_rows, n_classes))
  
  i = start
  if end == None: 
    end = n_rows
  
  with open(csv_path.format(case), 'r') as file:
    for line in file:
      if i == n_rows:
        break
      label = label_helper(line, n_classes, imap)
      labels[i] = label
      if label[0] == 1 and np.random.uniform() < 0.15:
        #obs[i] = np.random.normal(size=n_features)
        obs[i] = np.random.uniform(low=-100, high=100, size=n_features)
      else:
        obs[i] = obs_helper(line)
      i += 1


  if doscale:
    print('scaling')
    if case == 'train':
      with open('scaling.py', 'w+') as f:
        print("import numpy as np\n", file=f)
        print(f"mu = np.{repr(np.mean(obs, axis=0))}\n", file=f)
        print(f"var = np.{repr(np.std(obs, axis=0))}\n", file=f)
    obs = scale(obs, axis=0)
  
  data = DataSet([obs, labels])
  return data

def get_data(classmap, max_gb=1., reuse=False,
  memory_friendly=False, data_path='', doscale=False, dest=''):
  return (load_csv('train', max_gb=max_gb, classmap=classmap,
            reuse=reuse, memory_friendly=memory_friendly, data_path=data_path,
            doscale=doscale, dest=dest), 
          load_csv('test', max_gb=max_gb, classmap=classmap,
            reuse=reuse, memory_friendly=memory_friendly, data_path=data_path,
            doscale=doscale, dest=dest))





