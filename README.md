# Speech/Non-Speech Detection

Summary of purpose here.

## Directory Structure

The followinig directory structure is assumed.

```bash
root  
│  
└───FSD Kaggle/  
│   │  
│   └───FSDKaggle2018.audio_train/  
│   │  
│   └───FSDKaggle2018.audio_test/  
│  
└───amicorpus/  
│  
└───ami_public_manual_1.6.2/  
│  
└───src/ 
```

Except for the `amicorpus` directory, it is assumed that the path from root to any train file contains the string 'train' and does not contain the string 'test'. Likewise, the path from root to any test file contains the string 'test' and does not contain the string 'train'.

The AMI corpus is available [here](http://groups.inf.ed.ac.uk/ami/download/) and the FSD Kaggle data set is found [here](https://zenodo.org/record/2552860#.Xr4OMRNKjGK).

## Requirements

Requirements are

- Tensorflow 1.15
- Numpy 1.18.3
- Librosa 0.7.2
- Scikit-Learn 0.22.2
- progress 1.5
- argparse 1.4.0

## Making the Data

To make a data set with 0.01 second window (alpha), 0.01 second hop length (beta), and 5 sub-windows

`python mfcc.py -a 0.01 -b 0.01 -i 5 --speech_src {X} --nonspeech_src {Y}`

where `X` is the name of the parent directory containing the speech audio and `Y` is the name of the parent directory containing the non-speech audio. For instance,  

`python mfcc.py -a 0.01 -b 0.01 -i 5 --speech_src amicorpus --nonspeech_src 'FSD Kaggle'`

This will create the files `root/datadir/speech_sources/amicorpus_train.csv` and `root/datadir/nonspeech_sources/FSD Kaggle_train.csv`. These files are likely to be large when using a brief window span and/or multiple sub-windows.

## Training a Model

To train a model, from the `modeling` directory,

`python main.py --speech_src {X} --nonspeech_src {Y}`

where `X` and `Y` are as before. For instance,

`python main.py --spech_src amicorpus --nonspeech_src 'FSD Kaggle'`

This will combine the sources `X` and `Y` into a single file `root/datadir/combined_train.csv`. By default, the classes will be balanced. When the model loads the data set into memory, it will downsample the data set to 1 gb prior to training. The optional `-r` argument can be set to true (i.e `-r True`) to reuse an existing combined csv if it exists (need to commit this change). This avoids the process of combining the two data sets.
