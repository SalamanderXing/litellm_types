from setuptools import setup, find_packages
import os


os.system("rm -rf build dist")

setup(
    name="litellm_types",
    version="0.0.6",
    packages=find_packages(),
    install_requires=[
        "pydantic",
        "litellm",
        "beartype",
    ],
)
