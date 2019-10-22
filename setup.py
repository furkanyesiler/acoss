# - * - coding: utf - 8 -
import sys
import imp
import os
from setuptools import setup, find_packages, Command
from setuptools.extension import Extension
from shutil import rmtree
try:
    from Cython.Build import cythonize
    import numpy as np
except ImportError:
    raise ImportError("Couldn't found any cython and numpy installation.")


version = imp.load_source('acoss._version', os.path.join('acoss', '_version.py'))

with open('README.md') as file:
    long_description = file.read()


extra_compile_args = ["-Ofast"]
extra_link_args = []


ext_modules = Extension(
    "pySeqAlign",
    sources=["acoss/benchmark/utils/alignment_tools/pySeqAlign.pyx"],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++"
)

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
    packages=find_packages(exclude=['_version.py']),
    license='AGPL3.0',
    setup_requires=['cython'],
    include_dirs=[np.get_include()],
    ext_modules=cythonize(ext_modules, language_level=3),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='audio music dsp musicinformationretireval coversongidentification',
    project_urls={
        'Source': 'https://github.com/furkanyesiler/acoss',
        'Tracker': 'https://github.com/furkanyesiler/acoss/issues',
    },
    install_requires=[
        'madmom',
        'numpy>=1.16.5',
        'pandas',
        'scipy==1.2.1',
        'scikit-learn==0.19.2',
        'deepdish',
        # 'librosa==0.6.1',
        'essentia',
    ],
    extras_require={
        'docs': [],
        'tests': []
    },
    cmdclass={},
)
