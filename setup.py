try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='SmilNN',
    version='0.0.1',
    packages=['SmilNN'],
    url='https://github.com/AlexHung780312/KerasLSTM',
    license='MIT',
    author='alex',
    author_email='alexhung@ntnu.edu.tw',
    description='NN models for Keras framework',
    install_requires = ["keras >= 1.0.1"]
)
