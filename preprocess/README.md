# preprocess module

```extractors.py```
```similarity.py```
```features.py```

## Setup

* Create a virtualenv with the requirements

```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt

```

* Update the global variables in ```local_config.py``` with your respective paths


## Usage

* Inside the virtualenv
```python
# check help flag for more details of command-line arguments
python extractors.py -h

# when using the cpu mode with 4 threads (for running feature extraction with n threads)
python extractors.py -c '/path/to/collections/textfiles/dir/' -p '/path/to/features/dir/'-m 'cpu' -d 'benchmark' -n 4

# when using the cluster mode (for running array jobs in cluster)
python extractors.py -c '1_collections.txt' -p '/path/to/features/dir/'-m 'cluster' -d 'whatisacover'
```

