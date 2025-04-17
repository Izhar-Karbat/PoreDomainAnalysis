from setuptools import setup, find_packages

setup(
    name="md_analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scipy",
        "mdanalysis",
        "jinja2",
        "python-pptx",
    ],
    entry_points={
        'console_scripts': [
            'md_analysis=md_analysis.main:main',
        ],
    },
) 