#!/bin/bash
set -e -x

# Compile wheels only with python3.6** versions
for PYBIN in /opt/python/*/bin; do
    if [[ "$PYBIN" == "/opt/python/cp36-cp36m/bin" ]];
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
    if [[ "$DOCKER_IMAGE" == "acorreya/acoss-builds:manylinux1_i686" ]];
    then
        "${PYBIN}/pip" install  --no-index -f /io/wheelhouse/acoss-0.0.1-cp36-cp36m-manylinux1_i686.whl
    else
        "${PYBIN}/pip" install  --no-index -f /io/wheelhouse/acoss-0.0.1-cp36-cp36m-manylinux1_x86_64.whl
    fi
    echo "Running tests"
    (cd "$HOME"; "${PYBIN}/python -c 'from acoss import coverid; from pySeqAlign import qmax'")
done