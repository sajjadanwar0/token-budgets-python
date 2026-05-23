from setuptools import setup, find_packages

setup(
    name="token-budgets-mypy",
    version="0.2.0",
    description="Production-grade Mypy plugin for affine Budget enforcement",
    packages=find_packages(),
    install_requires=["mypy>=1.8.0"],
    python_requires=">=3.10",
)