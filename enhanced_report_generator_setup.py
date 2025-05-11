from setuptools import setup, find_packages

setup(
    name="enhanced_report_generator",
    version="0.1.0",
    packages=find_packages(include=["enhanced_report_generator", "enhanced_report_generator.*"]),
    include_package_data=True,
    package_data={
        "enhanced_report_generator": [
            "templates/*.html",
            "assets/*.css",
            "assets/*.js",
            "config/*.csv",
            "config/*.yaml",
        ],
    },
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scipy",
        "jinja2",
    ],
    entry_points={
        "console_scripts": [
            "generate-report=enhanced_report_generator.main:main",
        ],
    },
    author="Claude AI",
    author_email="noreply@anthropic.com",
    description="Enhanced Report Generator for MD Simulation Analysis",
    long_description="A tool for generating comprehensive comparative reports from MD simulations analysis data.",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires='>=3.9',
)