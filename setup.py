from setuptools import setup, find_packages

setup(
    name="cupydle",

    version="0.1.0",

    author="Ponzoni Nelson",
    author_email="npcuadra@gmail.com",
    url="https://github.com/lerker/cupydle",

    packages=find_packages(),
    include_package_data=True,
    scripts=[],

    license="LICENSE",

    description="Framework of deep learning for CUDA",
    #long_description=open("README.md").read(),

    install_requires=open("requirements.txt").read().split()
)
