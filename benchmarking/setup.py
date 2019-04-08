import sys
import os
import platform

from setuptools import setup
from setuptools.extension import Extension

# Ensure Cython is installed before we even attempt to install pySeqAlign
try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except:
    print("You don't seem to have Cython installed. Please get a")
    print("copy from www.cython.org or install it with `pip install Cython`")
    sys.exit(1)

## Get version information from _version.py
import re
VERSIONFILE="_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

# Use README.md as the package long description  
with open('README.md') as f:
    long_description = f.read()

class CustomBuildExtCommand(build_ext):
    """ This extension command lets us not require numpy be installed before running pip install ripser 
        build_ext command for use when numpy headers are needed.
    """

    def run(self):
        # Import numpy here, only when headers are needed
        import numpy
        # Add numpy headers to include_dirs
        self.include_dirs.append(numpy.get_include())
        # Call original build_ext command
        build_ext.run(self)

extra_compile_args = ["-Ofast"]
extra_link_args = []


ext_modules = Extension(
    "pySeqAlign",
    sources=["pySeqAlign.pyx"],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++"
)


setup(
    name="pySeqAlign",
    version=verstr,
    description="A Lean Sequence Alignment Library for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Chris Tralie",
    author_email="chris.tralie@gmail.com",
    packages=['pySeqAlign'],
    ext_modules=cythonize(ext_modules),
    install_requires=[
        'Cython',
        'numpy',
        'matplotlib',
    ],
    cmdclass={'build_ext': CustomBuildExtCommand},
)
