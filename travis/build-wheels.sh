#!/bin/bash
set -e -x

yum install -y llvm-3.9.0-libs-3.9.0-7.el7.centos.alonid.x86_64.rpm

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