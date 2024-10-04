#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

with open("requirements.txt") as requires_file:
    requirements = requires_file.read().split("\n")


with open("requirements_test.txt") as requires_file:
    test_requirements = requires_file.read().split("\n")


setup(
    name="heron-model",
    use_scm_version={"local_scheme": "no-local-version"},
    setup_requires=["setuptools_scm"],
    description="Heron is a toolkit for making machine learning-based gravitational waveform models in python.",
    #long_description=readme,# + "\n\n" + history,
    author="Daniel Williams",
    author_email="daniel.williams@ligo.org",
    url="https://github.com/transientlunatic/heron",
    packages=[
        "heron",
    ],
    package_dir={"heron": "heron"},
    entry_points={
        "console_scripts": ["heron=heron.cli:heron"],
        "asimov.pipelines": [
            "heron=heron.asimov:Pipeline",
            "heron injection=heron.asimov:InjectionPipeline",
        ],
    },
    include_package_data=True,
    install_requires=requirements,
    license="ISCL",
    zip_safe=True,
    keywords="heron",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: ISC License (ISCL)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    test_suite="tests",
    # tests_require=test_requirements
)
