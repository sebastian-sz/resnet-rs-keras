from setuptools import setup

with open("README.md", encoding="utf-8") as f:
    long_description = "\n" + f.read()

setup(
    name="resnet-rs-keras",
    version="1.1",
    description="A Keras implementation of Resnet-RS models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sebastian-sz/resnet-rs-keras",
    author="Sebastian Szymanski",
    author_email="mocart15@gmail.com",
    license="Apache",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    install_requires=["packaging"],
    packages=["resnet_rs"],
)
