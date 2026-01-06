from setuptools import setup, find_packages

setup(
    name="geometry-learning",
    version="0.1.0",
    description="Learning Geometry on Fixed Topologies for Graph Neural Networks",
    author="Research Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.7.0",
    ],
    python_requires=">=3.9",
)