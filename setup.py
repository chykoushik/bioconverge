from setuptools import setup, find_packages

setup(
    name="bioconverge",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "lifelines",
        "umap-learn",
        "hdbscan",
        "requests",
        "plotly",
        "nbformat",
    ],
    python_requires=">=3.8",
)
