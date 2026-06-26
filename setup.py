from setuptools import setup, find_packages

setup(
    name="mas_cartesian_architect",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
        "scipy>=1.10.0",
    ],
    author="MAS Research Team",
    description="Multi-Agent System with Cartesian Architect for avoiding combinatorial explosion",
)