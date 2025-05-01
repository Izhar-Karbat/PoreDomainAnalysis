# setup.py
from setuptools import setup, find_packages

# Read version from the package's __init__ file
# This avoids specifying it in two places
version = {}
with open("pore_analysis/core/__init__.py") as fp:
    exec(fp.read(), version)

# Read the long description from README.md
try:
    with open('README.md', encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Suite for analyzing K+ channel molecular dynamics data." # Fallback


setup(
    name="pore_analysis",
    version=version['__version__'], # Read version dynamically
    packages=find_packages(),
    include_package_data=True,
    package_data={
        # If any package contains *.json or *.html files, include them:
        "pore_analysis": [
            "plots_dict.json",      # Include pore_analysis/plots_dict.json
            "templates/*.html",     # Include all .html files in pore_analysis/templates/
            "templates/_*.html",    # Include partial templates as well
        ],
        # If data files were in submodules, list them too, e.g.:
        # "pore_analysis.modules.some_module": ["data_file.txt"],
    },
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
            # Add linters (e.g., 'flake8', 'black') or other dev tools here if desired
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
    long_description=long_description,
    long_description_content_type="text/markdown", # Specify type for README
    url="https://github.com/Izhar-Karbat/PoreDomainAnalysis", # Optional: Link to repository
    classifiers=[ # Optional: PyPI classifiers
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9", # Be specific
        "License :: OSI Approved :: MIT License", # Choose your license
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Physics",
        # Add relevant classifiers
    ],
    python_requires='>=3.9', # Specify minimum Python version
)
