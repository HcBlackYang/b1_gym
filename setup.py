from setuptools import find_packages
from distutils.core import setup

setup(
    name='b1_gym',
    version='1.0.0',
    author='Blake Yang',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='e1373705@u.nus.edu',
    description='Rl Algorithms And Isaac Gym environments for B1 Robots',
    install_requires=['isaacgym',
                      'matplotlib']
)