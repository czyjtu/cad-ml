from setuptools import setup, find_packages


with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name='coronaryx',
    packages=find_packages(),
    version='0.1.0',
    install_requires=required
)
