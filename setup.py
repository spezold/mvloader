#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup

setup(
    name="mvloader",
    version="0.2.201807031736",
    description="load medical image volumes (3D)",
    long_description="See readme.md in the project folder or on the github page.",
    author="Simon Pezold",
    author_email="simon.pezold@unibas.ch",
    url="https://github.com/spezold/mvloader",
    packages=["mvloader"],
    license="MIT License",
    python_requires='>=3.5',
    install_requires=["nibabel", "pydicom", "pynrrd", "numpy"],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)
