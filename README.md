# Speech/Non-Speech/Music Detection

## Directory Structure

The followinig directory structure is assumed.

```bash
root  
│  
└───DCASE/
│  
└───fma_medium/
│  
└───amicorpus/  
│  
└───ami_public_manual_1.6.2/  
│  
└───src/
│  
└───datadir/
```

The AMI corpus is available [here](http://groups.inf.ed.ac.uk/ami/download/) and the DCASE data set is found [here](https://zenodo.org/record/1247102#.XuYav2pKh26), and the FMA data set [here](https://github.com/mdeff/fma).

Generally, these data sets can be replaced with any other data sets provided they are located in `root/` and contain raw audio in a format supported by Librosa or one of its backends.

## Requirements

Requirements are

- Tensorflow 1.15 (not required for making data)
- Numpy 1.18.3
- Librosa 0.7.2
- Scikit-Learn 0.22.2
- progress 1.5
- argparse 1.4.0
- PyYaml

## Making the Data

To make the full data set,

`python make_data.py -gb 20 --outsize 20 --yaml /path/to/yaml`

where a default `foo.yaml` is located in `/root/src/`. This will make a CSV of extracted MFCC features for each class in `/root/datadir/bar/` and then combine these into a single, class-balanced CSV of data at `/root/datadir/bar/combined_train.csv`. The full data set is quite large so be sure to have at least 40gb of available space before running the program. However, by default, the class-specific CSVs will be deleted leaving only the combined CSV which generally doesn't take more 20gb. The optional `-gb` and `--outsize` arguments can be used to truncate the data creation process if not enough space is available. The `-gb` argument determines the maximum size of each class-specific CSV and the `--outsize` argument determines the size of the final, combined result.
