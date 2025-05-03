# setup.py
from setuptools import setup, find_packages, Command
from setuptools.command.install import install
from setuptools.command.develop import develop
import os
import sys
import subprocess

# Define a version fallback in case the file can't be read
version = {'__version__': '1.0.0'}  # Default version

# Function to check if a directory is a git submodule
def is_submodule(path):
    """Check if the given directory is a git submodule."""
    # Check for .git file which indicates a submodule (not a directory, but a file)
    git_file = os.path.join(path, '.git')
    if os.path.isfile(git_file):
        return True
    # Also check for standard .git directory which indicates a full repo
    if os.path.isdir(git_file):
        return True
    return False

# Custom command mixin to check for submodules
class SubmoduleCheckMixin:
    """A mixin to check for submodules before installation."""
    
    def run(self):
        # Check if pore_analysis is a submodule
        if os.path.exists('pore_analysis') and is_submodule('pore_analysis'):
            print("\n" + "="*80)
            print("WARNING: The pore_analysis directory appears to be a Git submodule.")
            print("This can cause issues with installation and package import.")
            print("")
            print("Options to resolve this:")
            print("1. Remove the submodule and incorporate the files directly:")
            print("   git rm --cached pore_analysis")
            print("   rm -rf pore_analysis/.git")
            print("   git add pore_analysis/")
            print("")
            print("2. For development, use: pip install -e .")
            print("   to ensure the local directory is used.")
            print("="*80 + "\n")
            
            # Try to ensure files are accessible
            submodule_init_needed = False
            if not os.path.exists('pore_analysis/core/__init__.py'):
                submodule_init_needed = True
            
            if submodule_init_needed:
                print("Attempting to initialize submodule to make files accessible...")
                try:
                    subprocess.check_call(['git', 'submodule', 'update', '--init', '--recursive'])
                    print("Submodule initialized successfully.")
                except (subprocess.SubprocessError, FileNotFoundError) as e:
                    print(f"Failed to initialize submodule: {e}")
                    print("Installation may fail if required files aren't accessible.")
        
        # Call the parent class run method to continue with installation
        super().run()

# Create the custom command classes
class InstallWithCheck(SubmoduleCheckMixin, install):
    """Custom install command that checks for submodules."""
    pass

class DevelopWithCheck(SubmoduleCheckMixin, develop):
    """Custom develop command that checks for submodules."""
    pass

# Try reading version from the package's __init__ file if it exists
try:
    version_file_path = "pore_analysis/core/__init__.py"
    if os.path.exists(version_file_path):
        with open(version_file_path) as fp:
            exec(fp.read(), version)
    else:
        print(f"Warning: {version_file_path} not found. Using default version.", file=sys.stderr)
except Exception as e:
    print(f"Warning: Could not read version from {version_file_path}: {e}", file=sys.stderr)

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
    packages=find_packages(exclude=["tests"]),
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
    # Add custom commands to check for and warn about submodule issues
    cmdclass={
        'install': InstallWithCheck,
        'develop': DevelopWithCheck,
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