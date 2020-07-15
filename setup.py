from __future__ import print_function
import os.path
import sys
import setuptools
from numpy.distutils.core import setup

try:
    import numpy
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)


DISTNAME = 'featurelowrank'
DESCRIPTION = "A polylearn based Feature Low Rank Model Implementation in Python."
VERSION = '0.0.dev0'


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)

    config.add_subpackage('featurelowrank')

    return config


if __name__ == '__main__':
    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))

    os.chdir(local_path)
    sys.path.insert(0, local_path)
    setup(configuration=configuration,
          name=DISTNAME,
          maintainer='Kyohei Atarashi',
          include_package_data=True,
          version=VERSION,
          zip_safe=False,  # the package can run out of an .egg file
          )
