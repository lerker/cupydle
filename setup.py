#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="cupydle",
    version="1.0.0",
    author="Ponzoni Nelson",
    author_email="npcuadra@gmail.com",
    url="https://github.com/lerker/cupydle",
    packages=find_packages(),
    include_package_data=True,
    scripts=[],
    license="LICENSE",
    description="Framework of deep learning GP-GPU",
    #long_description=open("README.md").read(),
    install_requires=open("requirements.txt").read().split()
)
