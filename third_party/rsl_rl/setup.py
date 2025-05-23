from setuptools import setup, find_packages

setup(name='rsl_rl',
      version='2.0.0',
      author='Blake Yang',
      author_email='e1373705@u.nus.edu',
      license="BSD-3-Clause",
      packages=find_packages(),
      description='Fast and simple RL algorithms implemented in pytorch',
      python_requires='>=3.6',
      install_requires=[
            "torch>=1.4.0",
            "torchvision>=0.5.0",
            "numpy>=1.16.4"
      ],
      )
