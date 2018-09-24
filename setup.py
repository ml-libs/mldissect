import os
import re
import sys
from setuptools import setup, find_packages


PY_VER = sys.version_info

if not PY_VER >= (3, 6):
    raise RuntimeError('mldissect does not support Python earlier than 3.6')


def read(f):
    return open(os.path.join(os.path.dirname(__file__), f)).read().strip()


install_requires = [
    'numpy',
    'terminaltables',
]
extras_require = {}


def read_version():
    regexp = re.compile(r"^__version__\W*=\W*'([\d.abrc]+)'")
    init_py = os.path.join(os.path.dirname(__file__),
                           'mldissect', '__init__.py')
    with open(init_py) as f:
        for line in f:
            match = regexp.match(line)
            if match is not None:
                return match.group(1)
        else:
            msg = 'Cannot find version in mldissect/__init__.py'
            raise RuntimeError(msg)


classifiers = [
    'License :: OSI Approved :: Apache Software License',
    'Intended Audience :: Developers',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Operating System :: POSIX',
    'Development Status :: 2 - Pre-Alpha',
    'Framework :: AsyncIO',
]


setup(name='mldissect',
      version=read_version(),
      description=('mldissect - model agnostic explanations'),
      long_description='\n\n'.join((read('README.rst'), read('CHANGES.txt'))),
      install_requires=install_requires,
      classifiers=classifiers,
      platforms=['POSIX'],
      author='Nikolay Novik',
      author_email='nickolainovik@gmail.com',
      url='https://github.com/ml-libs/mldissect',
      download_url='https://pypi.python.org/pypi/mldissect',
      license='Apache 2',
      packages=find_packages(),
      extras_require=extras_require,
      keywords=['mldissect', 'model explanation'],
      zip_safe=True,
      include_package_data=True)
