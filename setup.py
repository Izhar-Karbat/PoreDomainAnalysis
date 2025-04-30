# setup.py
from setuptools import setup, find_packages

# Read version from the package's __init__ file
# This avoids specifying it in two places
version = {}
with open("pore_analysis/core/__init__.py") as fp:
    exec(fp.read(), version)

setup(
    name="pore_analysis",
    version=version['__version__'], # Read version dynamically
    packages=find_packages(),
    # List direct dependencies required to run the core analysis package
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scipy",
        "scikit-learn",
        "mdanalysis>=2.0.0", # Specify minimum version if known
        "jinja2",
        "tqdm", # Added dependency for progress bars
    ],
    # Suggest extra dependencies for development/testing
    extras_require={
        'dev': [
            'pytest',
            'Pillow', # Needed for some tests that create dummy images
            # Add other dev tools like black, flake8, etc. here
        ]
    },
    entry_points={
        'console_scripts': [
            'pore_analysis=pore_analysis.main:main',
        ],
    },
    # Add other metadata as needed
    author="Izhar Karbat", # Optional: Add author info
    author_email="izhar.karbat@weizmann.ac.il", # Optional
    description="Suite for analyzing K+ channel molecular dynamics data.", # Optional
    long_description=open('README.md').read(), # Optional: Use README
    long_description_content_type="text/markdown", # Optional
    url="https://github.com/Izhar-Karbat/PoreDomainAnalysis", # Optional: Link to repository
    classifiers=[ # Optional: PyPI classifiers
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # Choose your license
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Molecular Dynamics",
        "Topic :: Scientific/Engineering :: Biophysics",
    ],
    python_requires='>=3.9', # Specify minimum Python version
)