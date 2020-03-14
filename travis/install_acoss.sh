#!/bin/bash
set -e -x

pip3 install --upgrade pip

pip3 install -r dev_requirements.txt

cd ../ && python3 setup.py install
