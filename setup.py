from setuptools import setup, find_packages

install_requires = ["numpy"]

setup(name="needle",
      version='0.0.1',
      install_requires=install_requires,
      packages=find_packages())