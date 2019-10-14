#!/bin/bash
set -e -x

# Use Python3.6. CentOS 5's native python is too old...
PYBIN=/opt/python/cp36-cp36m/bin/

# Compile wheels only with python3.6** versions
for PYBIN in /opt/python/*/bin; do
    if [[ "$PYBIN" == *"cp36m"* ]];
    then
        "${PYBIN}/pip" install -r /io/travis/dev-requirements.txt
        "${PYBIN}/pip" wheel /io/ -w wheelhouse/
    else
        echo "$PYBIN"
    fi
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
# Do not run auditwheel for six package because of a bug
# https://github.com/pypa/python-manylinux-demo/issues/7
    if [[ "$whl" != wheelhouse/six* ]];
    then
        auditwheel repair "$whl" -w /io/wheelhouse/
    else
        cp "$whl" /io/wheelhouse/
    fi
done

# Install packages and test
for PYBIN in /opt/python/*/bin/; do
    if [[ "$PYBIN" == *"cp36m"* ]];
    then
        "${PYBIN}/pip" install acoss --no-index -f /io/wheelhouse
        (cd "$HOME"; "${PYBIN}/python -c 'from acoss import coverid; from pySeqAlign import qmax'")
    else
        echo "$PYBIN"
    fi
done
