from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='qcoptim',
    version='0.1',
    description='',
    long_description=long_description,
    url='',
    author='Frederic Sauvage, Kiran Khosla, Chris Self',
    classifiers=[],
    keywords='',
    packages=find_packages(exclude=['docs', 'tests']),
    install_requires=[
        'numpy',
        'tensorflow',
        'qiskit',
    ],
    extras_require={  # Optional
    },
    project_urls={},
)
