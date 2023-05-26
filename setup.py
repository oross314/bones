from setuptools import setup

setup(
    name='bones',
    version='1.0',
    author='Olive Ross',
    author_email='ogr8@cornell.edu',
    description='A user friendly interface for implementing and training a deep neural network in pytorch',
    packages=['bones'],
    install_requires=[
        'pandas',
        'torch',
        'copy',
        'numpy',
        'matplotlib'
        # Add any other dependencies here
    ]
)