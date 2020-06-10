

'''
Libri: 0.11080034000473364 0.910935115239116 0.881619535961162

Vox: 0.2466148250915429 0.8346169671231234 0.6480534598151981

Flickr: 0.15383308260906506 0.8773783404219164 0.7974880926796843
'''

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str)
args = parser.parse_args()

if args.path:
	subdir = args.path + '/'
else:
	subdir = ''

with open('./tf_model.py', 'w+') as m, \
     open(f'./weights/{subdir}feedforward.dense.kernel:0.txt', 'r') as k1, \
     open(f'./weights/{subdir}feedforward.dense.bias:0.txt', 'r') as b1, \
     open(f'./weights/{subdir}feedforward.logits.kernel:0.txt', 'r') as k2, \
     open(f'./weights/{subdir}feedforward.logits.bias:0.txt', 'r') as b2:
  
  n_input_features = 500
  m.write('import numpy as np\n\n')
  
  m.write('hk = np.array([\n')
  for i, row in enumerate(k1):
    if i != 0:
      m.write(',\n')
    m.write('  [' + row.strip() + ']')
  m.write('\n])\n\n')

  m.write('hb = np.array([\n')
  for i, row in enumerate(b1):
    if i != 0:
      m.write(',\n')
    m.write('  [' + row.strip() + ']')
  m.write('\n])\n\n')

  m.write('ok = np.array([\n')
  for i, row in enumerate(k2):
    if i != 0:
      m.write(',\n')
    m.write('  [' + row.strip() + ']')
  m.write('\n])\n\n')

  m.write('ob = np.array([\n')
  for i, row in enumerate(b2):
    if i != 0:
      m.write(',\n')
    m.write('  [' + row.strip() + ']')
  m.write('\n])\n\n')

  m.write('def classify(x):\n')
  m.write('  return np.argmax(np.add(np.matmul(np.maximum(np.add(np.matmul(x, hk.T), hb.T), 0), ok.T), ob.T), axis=1)\n\n')
  #m.write('  return np.maximum(np.sign(np.add(np.matmul(np.maximum(np.add(np.matmul(x, hk.T), hb.T), 0), ok.T), ob)), 0)\n\n')

  m.write('def tf_classify(x):\n')
  m.write('  return classify(x)')

'''


  m.write('def tf_classify(x):\n')
  inputs = []
  
  for idx, (_k, _b) in enumerate(((k1, b1), (k2, b2))):
    
    n_neurons = 0
    outputs = []
    vname = 'h' if idx == 0 else 'o'
    
    if len(inputs) == 0:
      for j in range(n_input_features):
        inputs.append(f'float(x[{j}])')
    
    for i, (line, b) in enumerate(zip(_k, _b)):
      n_neurons += 1
      outputs.append(f'{vname}_{i}')
      
      if idx == 0:
        m.write(f'  {vname}_{i} = max(')
      else:
        m.write(f'  {vname}_{i} = ')      
      
      for j, w in enumerate(line.strip().split(',')):
        m.write(f'{w}*{inputs[j]} + ')
      
      if idx == 0:
        m.write(f'{b.strip()}, 0.)\n')
      else:
        m.write(f'{b.strip()}\n')
      
    inputs = outputs
  
  if len(outputs) == 1:
    m.write('  return 1 if o_0 > 0 else 0')
  elif len(outputs) == 2:
    m.write('  return 0 if o_0 > o_1 else 1')
  else:
  	m.write('  return np.argmax(outputs)')
  m.write('\n\n')

  m.write('def classify(x):\n')
  m.write('  return np.apply_along_axis(tf_classify, 1, x)')

'''