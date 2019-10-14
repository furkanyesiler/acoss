#!/bin/bash
set -e -x

# Compile wheels
#for PYBIN in /opt/python/*/bin; do
#    "${PYBIN}/pip" install -r /io/travis/dev-requirements.txt
#    "${PYBIN}/pip" wheel /io/ -w wheelhouse/
#done
PYBIN=python3.6
"${PYBIN}/pip" install -r /io/travis/dev-requirements.txt
"${PYBIN}/pip" wheel /io/ -w wheelhouse/

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
    "${PYBIN}/pip" install acoss --no-index -f /io/wheelhouse
    (cd "$HOME"; "${PYBIN}/python -c 'from acoss import features; from pySeqAlign import qmax'")
done