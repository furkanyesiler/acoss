# acoss: Audio Cover Song Suite
[![Build Status](https://travis-ci.org/furkanyesiler/acoss.svg?branch=master)](https://travis-ci.org/furkanyesiler/acoss)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

[acoss: Audio Cover Song Suite](https://github.com/furkanyesiler/acoss) is a feature extraction and benchmarking frameworks for the 
cover song identification (CSI) tasks. This tool has been developed along with the new [DA-TACOS](https://mtg.github.io/da-tacos) dataset. 


`acoss` includes a standard feature extraction framework with state-of-art audio features for CSI task and open source implementations of seven state-of-the-art CSI algorithms to facilitate the future work in this
line of research. Using this framework, researchers can easily compare existing algorithms on different datasets,
and we encourage all CSI researchers to incorporate their
algorithms into this framework which can easily done following the usage examples. 

Please site our paper if you use this tool in your resarch.

> [Furkan Yesiler, Chris Tralie, Albin Correya, Diego F. Silva, Philip Tovstogan, Emilia Gómez, and Xavier Serra. Da-TACOS: A Dataset for Cover Song Identification and Understanding. In 20th International Society for Music Information Retrieval Conference (ISMIR 2019), Delft, The Netherlands, 2019.](https://repositori.upf.edu/bitstream/handle/10230/42771/yesiler_ismir19_datacos.pdf?sequence=1&isAllowed=y)


Benchmarking results on [Da-Tacos](https://mtg.github.io/da-tacos) dataset can be found in the paper.

## Setup & Installation

We recommend you to install the package inside a python [virtualenv](https://docs.python.org/3/tutorial/venv.html). 


#### Install using pip 

```bash
pip install acoss
```

OR

#### Install from source (recommended)

- Clone or download the repo.
- Install `acoss` package by using the following command inside the directory.
```bash
python3 setup.py install
```


#### Additional dependencies

`acoss` requires a local installation of madmom library for computing some audio features and essentia library for similarity algorithms. 

For linux-based distro users,
```
pip install "acoss[extra-deps]"
```

or if you are a Mac OSX user, you can install `essentia` from homebrew 
```
brew tap MTG/essentia
brew install essentia --HEAD
```

## Documentation and examples

[`acoss`](/) mainly provides the following python sub-modules-

- `acoss.algorithms`: Sub-module with various cover identification algorithms, utilities for similarity comparison and an template for adding new algorithms. 

- `acoss.coverid`: Interface to benchmark a specific cover identification algorithms.

- `acoss.features`: Sub-module with implementation of various audio features.

- `acoss.extractors` : Interface to do efficient batch audio feature extraction for an audio dataset.

- `acoss.utils` : Utility functions used in acoss package.

### [`acoss.coverid.benchmark`]()

| Cover Song Identification algorithms in `acoss`  |  |    
|---|---|
|  `Serra09`  | [Paper](https://iopscience.iop.org/article/10.1088/1367-2630/11/9/093017)  | 
|  `LateFusionChen`  | [Paper](https://link.springer.com/article/10.1186/s13636-017-0108-2)  | 
|  `EarlyFusionTraile`  | [Paper](https://arxiv.org/pdf/1707.04680.pdf)  | 
|  `SiMPle`  | [Paper](https://www.cs.ucr.edu/~eamonn/MP_Music_ISMIR.pdf)  | 
|  `FTM2D`  | [Paper](https://academiccommons.columbia.edu/doi/10.7916/D8Z60ZBV)  | 
|  `MOVE`  | adding soon ...  | 


### [`acoss.extractors.batch_feature_extractor`]()
```json
{
	"chroma_cens": numpy.ndarray,
	"crema": numpy.ndarray,
	"hpcp": numpy.ndarray,
	"key_extractor": {
		"key": numpy.str_,
		"scale": numpy.str_,_
		"strength": numpy.float64
	},
	"madmom_features": {
		"novfn": numpy.ndarray, 
		"onsets": numpy.ndarray,
		"snovfn": numpy.ndarray,
		"tempos": numpy.ndarray
	}
	"mfcc_htk": numpy.ndarray,
	"label": numpy.str_,
	"track_id": numpy.str_
}
```

### Dataset structure required for acoss


#### Audio data

```
audio_dir
    /work_id
        /track_id.mp3
```   

#### Feature file

```
feature_dir
    /work_id
        /track_id.h5 
```   

```python
import deepdish as dd

feature = dd.load("feature_file.h5")
```

An example feature file will be in the following structure.

 ```json
{
    'feature_1': [],
    'feature_2': [],
    'feature_3': {'type_1': [], 'type_2': [], ...},
    ......  
}
```

#### CSV annotation file for a dataset


| work_id  | track_id  |  
|---|---|
|  `W_163930` | `P_546633`  | 
|  ... | ...  | 


`acoss` benchmark methods expect the dataset annotation csv file in the above given format.


There are also some utility functions in `acoss` which generates the csv annotation file automatically for da-tacos from it's subset metadata file and for covers80 dataset from it's audio data directory.
```python
from acoss.utils import da_tacos_metadata_to_acoss_csv
da_tacos_metadata_to_acoss_csv("da-tacos_benchmark_subset_metadata.json", 
                            "da-tacos_benchmark_subset.csv")


from acoss.utils import generate_covers80_acoss_csv
generate_covers80_acoss_csv("coversongs/covers32k/", 
                            "covers80_annotations.csv")
```



For quick prototyping, let's use the tiny [covers80](), dataset.

### Audio feature extraction

```python
from acoss.utils import COVERS_80_CSV
from acoss.extractors import batch_feature_extractor
from acoss.extractors import PROFILE

print(PROFILE)
```


> {
    'sample_rate': 44100,
    'input_audio_format': '.mp3',
    'downsample_audio': False,
    'downsample_factor': 2,
    'endtime': None,
    'features': ['hpcp',
        'key_extractor',
        'madmom_features',
        'mfcc_htk']
}


Compute

```python
# Let's define a custom acoss extractor profile
extractor_profile = {
           'sample_rate': 32000,
           'input_audio_format': '.mp3',
           'downsample_audio': True,
           'downsample_factor': 2,
           'endtime': None,
           'features': ['hpcp']
}

# path to audio data
audio_dir = "../coversongs/covers32k/"
# path where you wanna store your data
feature_dir = "features/"

# Run the batch feature extractor with 4 parallel workers
batch_feature_extractor(dataset_csv=COVERS_80_CSV, 
                        audio_dir=audio_dir, 
                        feature_dir=feature_dir,
                        n_workers=4,
                        mode="parallel", 
                        params=extractor_profile)
```

### Benchmarking

```python
from acoss.coverid import benchmark, algorithm_names
from acoss.utils import COVERS_80_CSV

# path to where your audio feature h5 files are located
feature_dir = "features/"

# list all the available algorithms in acoss 
print(algorithm_names)

# we can easily benchmark any of the given cover id algorithm
# on the given dataset using the following function
benchmark(dataset_csv=COVERS_80_CSV, 
        feature_dir=feature_dir,
        algorithm="Serra09", 
        parallel=False)
# result of the evaluation will be stored in a csv file 
# in the current working directory.
```

## How to add my algorithm in `acoss`?

### Defining my algorithm
```python
from acoss.algorithms.algorithm_template import CoverSimilarity

class MyCoverIDAlgorithm(CoverSimilarity):
    def __init__(self, 
                dataset_csv, 
                datapath, 
                name="MyAlgorithm", 
                shortname="mca"):
        CoverAlgorithm.__init__(self, 
                                dataset_csv=dataset_csv, 
                                name=name, 
                                datapath=datapath, 
                                shortname=shortname)

    def load_features(self, i):
        """Define how to want to load the features"""
        feats = CoverAlgorithm.load_features(self, i)
        # add your modification to feature arrays
        return

    def similarity(self, idxs):
        """Define how you want to compute the cover song similarity"""
        return

```

### Running benchmark on my algorithm
```python
# create an instance your algorithm
my_awesome_algorithm = MyCoverIDAlgorithm(dataset_csv, datapath)

# run pairwise comparison
my_awesome_algorithm.all_pairwise()

# Compute standard evaluation metrics
for similarity_type in my_awesome_algorithm.Ds.keys():
        print(similarity_type)
        my_awesome_algorithm.getEvalStatistics(similarity_type)

my_awesome_algorithm.cleanup_memmap()
```



## How to contribute?

* Fork the repo!
* Create your feature branch: git checkout -b my-new-feature
* Add your new audio feature or cover identification algorithm to acoss.
* Commit your changes: git commit -am 'Add some feature'
* Push to the branch: git push origin my-new-feature
* Submit a pull request


## Acknowledgments

This project has received funding from the European Union's Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 765068 (MIP-Frontiers).

This work has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No 770376 (Trompa).

<img src="https://upload.wikimedia.org/wikipedia/commons/b/b7/Flag_of_Europe.svg" height="64" hspace="20">

