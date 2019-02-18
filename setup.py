
from setuptools import setup, find_packages

import sys


setup(name='mappertools',
      author='Emerson G. Escolar',
      version='0.2',
      description='Mapper tools',
      packages=find_packages(exclude=["*.tests"]),
      install_requires=['numpy', 'matplotlib', 'networkx']
)
