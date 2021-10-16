from pkg_resources import DistributionNotFound, get_distribution
from setuptools import setup

TF_MIN_VERSION = 2.2
TF_MAX_VERSION = 2.6

with open("README.md", encoding="utf-8") as f:
    long_description = "\n" + f.read()


def _package_exists(name: str) -> bool:
    """Check whether package is present in the system."""
    try:
        get_distribution(name)
    except DistributionNotFound:
        return False
    else:
        return True


def _get_tensorflow_requirement():
    """Avoid re-download and misdetection of package."""
    if _package_exists("tensorflow-cpu"):
        return [f"tensorflow-cpu>={TF_MIN_VERSION},<={TF_MAX_VERSION}"]
    elif _package_exists("tensorflow-gpu"):
        return [f"tensorflow-gpu>={TF_MIN_VERSION},<={TF_MAX_VERSION}"]
    else:
        return [f"tensorflow>={TF_MIN_VERSION},<={TF_MAX_VERSION}"]


setup(
    name="resnet-rs-keras",
    version="1.0",
    description="A Keras implementation of Resnet-RS models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sebastian-sz/resnet-rs-keras",
    author="Sebastian Szymanski",
    author_email="mocart15@gmail.com",
    license="Apache",
    python_requires=">=3.6.0,<3.10",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    #    packages=["resnet_rs"],  TODO
    install_requires=_get_tensorflow_requirement(),
)
