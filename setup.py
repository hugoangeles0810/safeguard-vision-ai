"""
Setup script for fall detection package.

Install in development mode with: pip install -e .
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="safeguard-vision-ai",
    version="0.1.0",
    author="Fall Detection Team",
    description="Fall detection system using pose estimation and temporal classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hugoangeles0810/safeguard-vision-ai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "mediapipe>=0.10.0",
        "ultralytics>=8.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tensorboard>=2.13.0",
        "opencv-python>=4.8.0",
        "imageio>=2.31.0",
        "gradio>=4.0.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "pytest>=7.4.0",
        ],
    },
)
