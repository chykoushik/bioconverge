from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="bioconverge",
    version="0.1.4",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
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