#!/usr/bin/env python
# -*- coding: utf-8 -*-
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open("requirements.txt") as requires_file:
    requirements = requires_file.read().split("\n")


with open("requirements_test.txt") as requires_file:
    test_requirements = requires_file.read().split("\n")




setup(
    name='heron-model',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    description="Heron is a machine learning package for Python",
    long_description=readme + '\n\n' + history,
    author="Daniel Williams",
    author_email='daniel.williams@ligo.org',
    url='https://github.com/transientlunatic/heron',
    packages=[
        'heron',
    ],
    package_dir={'heron': 'heron'},
    include_package_data=True,
    install_requires=requirements,
    license="ISCL",
    zip_safe=True,
    keywords='heron',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: ISC License (ISCL)',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    test_suite='tests',
    #tests_require=test_requirements
)
