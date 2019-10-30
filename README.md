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

> Furkan Yesiler, Chris Tralie, Albin Correya, Diego F. Silva, Philip Tovstogan, Emilia Gómez, and Xavier Serra. Da-TACOS: A Dataset for Cover Song Identification and Understanding. In 20th International Society for Music Information Retrieval Conference (ISMIR 2019), Delft, The Netherlands, 2019.


## Setup & Installation

We recommend you to install the package inside a python [virtualenv](https://docs.python.org/3/tutorial/venv.html). 

#### Install from source (recommended)

- Clone or download the repo.
- Install `acoss` package by using the following command inside the directory.
```bash
python3 setup.py install
```

OR

#### Install using pip (currenly only of linux distros)

```bash
pip install acoss
```


> NOTE: You might need to have a local installation of [librosa](https://librosa.github.io/librosa/0.6.1/index.html)
python library and other optional dependencies.

## Usages

The usage examples will be updated soon here.

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

