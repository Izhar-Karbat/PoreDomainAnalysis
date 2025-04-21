from setuptools import setup, find_packages

setup(
    name="pore_analysis",
    version="1.6.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scipy",
        "scikit-learn",
        "mdanalysis",
        "jinja2",
    ],
    entry_points={
        'console_scripts': [
            'pore_analysis=pore_analysis.main:main',
        ],
    },
) 