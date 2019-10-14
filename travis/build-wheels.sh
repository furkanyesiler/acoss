#!/bin/bash
set -e -x

# Build tools for using original quay.io/pypa/manylinux1_* docker images
/io/travis/build_tools.sh

# Compile wheels
for PYBIN in /opt/python/*/bin; do
    "${PYBIN}/pip" install -r /io/travis/dev-requirements.txt
    "${PYBIN}/pip" wheel /io/ -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    auditwheel repair "$whl" --plat $PLAT -w /io/wheelhouse/
done

# Install packages and test
for PYBIN in /opt/python/*/bin/; do
    "${PYBIN}/pip" install acoss --no-index -f /io/wheelhouse
    (cd "$HOME"; "${PYBIN}/python -c 'from acoss import features; from pySeqAlign import qmax'")
done