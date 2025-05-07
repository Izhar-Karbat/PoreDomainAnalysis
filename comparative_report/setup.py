#!/usr/bin/env python3
"""
Setup script for the pore_cross_analysis package.
"""

from setuptools import setup, find_packages
import os

# Read the long description from the README file
readme_content = """
# Pore Cross-Analysis Suite

A comprehensive toolkit for comparative analysis of toxin-bound and control ion channel systems from molecular dynamics simulations.

## Overview

The Pore Cross-Analysis Suite enables systematic comparisons between toxin-bound and control systems by:

1. Extracting metrics from individual system databases
2. Performing statistical comparisons between system types
3. Generating visualizations of key differences
4. Providing AI-powered insights (optional)
5. Creating comprehensive HTML reports

"""

setup(
    name="pore_cross_analysis",
    version="0.1.0",
    description="Cross-Analysis Suite for toxin-bound and control ion channel systems",
    long_description=readme_content,
    long_description_content_type="text/markdown",
    author="",
    author_email="",
    url="",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'cross_analysis': ['templates/*.html'],
    },
    scripts=['cross_analysis_main.py'],
    install_requires=[
        'numpy>=1.20.0',
        'pandas>=1.2.0',
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0',
        'jinja2>=3.0.0',
        'markdown>=3.3.0',
        'scipy>=1.6.0',
        'statsmodels>=0.12.0',
        'requests>=2.25.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.10.0',
            'flake8>=3.8.0',
            'black>=21.5b0',
        ],
    },
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    keywords='molecular dynamics, ion channels, toxin, analysis',
)