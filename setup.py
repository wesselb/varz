from setuptools import find_packages, setup

requirements = [
    "numpy>=1.16",
    "scipy>=1.3",
    "plum-dispatch>=1",
    "backends>=1",
    "wbml>=0.3",
]

setup(
    packages=find_packages(exclude=["docs"]),
    python_requires=">=3.6",
    install_requires=requirements,
    include_package_data=True,
)
