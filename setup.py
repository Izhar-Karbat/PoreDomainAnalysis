# setup.py
from setuptools import setup, find_packages

# Read version from the package's __init__ file
version = {}
with open("pore_analysis/core/__init__.py") as fp:
    exec(fp.read(), version)

# Read the long description from README.md
try:
    with open('README.md', encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Suite for analyzing K+ channel molecular dynamics data." # Fallback

# Base requirements for all analysis modules EXCEPT pocket_analysis
base_requires = [
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "scipy", # Now needed for some stats (e.g., skewness)
    "scikit-learn", # Needed by DWGate / Pocket Analysis imports
    "mdanalysis>=2.0.0",
    "jinja2",
    "tqdm",
]

# Extra requirements specifically for pocket_analysis
pocket_requires = [
    "torch",
    "torch_geometric", # Often needed with torchmd-net
    "torchmd-net", # Add torchmd-net dependency
    # Add any other specific dependencies identified during Train_ET integration
]

setup(
    name="pore_analysis",
    version=version['__version__'],
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "pore_analysis": [
            "plots_dict.json",
            "templates/*.html",
            "templates/_*.html",
            # Include ML model files
            "modules/pocket_analysis/ml_model/*.json",
            "modules/pocket_analysis/ml_model/*.pth",
        ],
    },
    # Base requirements
    install_requires=base_requires,
    # Optional dependencies grouped by feature
    extras_require={
        'pocket': pocket_requires, # Requires GPU/CUDA + complex setup
        'dev': [ # Development/testing tools
            'pytest',
            'Pillow',
            # Optional: Add linters like 'flake8', 'black' here
        ],
        # Define an 'all' option to install base + pocket easily
        'all': pocket_requires,
    },
    entry_points={
        'console_scripts': [
            'pore_analysis=pore_analysis.main:main',
        ],
    },
    author="Izhar Karbat",
    author_email="izhar.karbat@weizmann.ac.il",
    description="Suite for analyzing K+ channel molecular dynamics data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Izhar-Karbat/PoreDomainAnalysis",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires='>=3.9',
)