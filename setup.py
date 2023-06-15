from setuptools import setup, find_packages

setup(
    name='torchbones',
    version='1.2.6',
    author='Olive Ross',
    author_email='ogr8@cornell.edu',
    description='A user friendly interface for implementing and training a deep neural network in pytorch',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'torch',
        'numpy',
        'matplotlib',
        'scikit-learn'
        # Add any other dependencies here
    ]
)