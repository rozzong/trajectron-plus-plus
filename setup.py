from setuptools import find_packages, setup

from trajectron_plus_plus import __version__


with open("README.md") as f:
    long_description = f.read()

required = []
with open("requirements.txt") as f:
    for package in f.readlines():
        required.append(package.strip())


setup(
    name="trajectron-plus-plus",
    version=__version__,
    description="A PyTorch implementation of Trajectron++",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Tim Salzmann, Boris Ivanovic, Punarjay Chakravarty, and Marco Pavone",
    maintainer="Gabriel Rozzonelli",
    maintainer_email="gabriel.rozzonelli@skoltech.ru",
    url="https://github.com/rozzong/trajectron-plus-plus",
    packages=find_packages(),
    install_requires=required,
    python_requires=">=3.8.0",
)
