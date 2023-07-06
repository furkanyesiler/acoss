# - * - coding: utf - 8 -
import sys
import imp
import os
from setuptools import setup, find_packages

version = imp.load_source('acoss._version', os.path.join('acoss', '_version.py'))

with open('README.md') as file:
    long_description = file.read()

setup(
    name='acoss',
    version=version.version,
    description='Audio Cover Song Suite (acoss): A feature extraction and '
                'benchmarking suite for cover song identification tasks',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/furkanyesiler/acoss',
    author='Albin Correya, Furkan Yesiler, Chris Traile, Philip Tovstogan, and Diego Silva',
    author_email='albin.correya@upf.edu',
    packages=find_packages(),
    license='AGPL3.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='audio music dsp musicinformationretireval coversongidentification',
    project_urls={
        'Source': 'https://github.com/furkanyesiler/acoss',
        'Tracker': 'https://github.com/furkanyesiler/acoss/issues',
    },
    include_package_data=True,
    package_data={'': ['data/*.csv']},
    install_requires=[
        'numpy>=1.16.5',
        'numba>=0.43.0',
        'pandas>=0.25.0',
        'scipy==1.10.0',
        'scikit-learn==0.19.2',
        'deepdish>=0.3.6',
        'librosa==0.6.1',
        'progress>=1.5'
    ],
    extras_require={
        'docs': [],
        'tests': [],
        'extra-deps': ['essentia', 
                      'madmom>=0.16.1'],
        'machine_learning': []
    },
    cmdclass={},
)   
