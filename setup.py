from setuptools import setup, find_packages

setup(
    name="emission-forecast",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "requests>=2.31.0",
        "autogluon>=1.2.0"
    ],
) 