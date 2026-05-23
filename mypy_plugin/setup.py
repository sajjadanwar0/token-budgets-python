from setuptools import setup

setup(
    name="token-budgets-mypy",
    version="0.2.0",
    description="Production-grade Mypy plugin for affine Budget enforcement",
    packages=["token_budgets_mypy"],
    package_dir={"token_budgets_mypy": "token_budgets_mypy"},
    install_requires=["mypy>=1.8.0"],
    python_requires=">=3.10",
    zip_safe=False,
)
