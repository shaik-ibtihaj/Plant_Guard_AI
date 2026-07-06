"""
Plant Guard AI - Package Setup
================================
Python package configuration for the Plant Guard AI project.
"""

from setuptools import setup, find_packages

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="plant-guard-ai",
    version="0.1.0",
    author="Plant Guard AI Team",
    author_email="contact@plantguardai.com",
    description="Intelligent Plant Disease Detection & Severity Assessment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/Plant_Guard_AI",
    packages=find_packages(exclude=["tests*", "notebooks*", "scripts*"]),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.1.0",
            "pytest-cov>=5.0.0",
            "black>=24.3.0",
            "flake8>=7.0.0",
            "isort>=5.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # TODO: Add CLI entry points
            # "plant-guard-train=ai.training.train:main",
            # "plant-guard-infer=ai.training.inference:main",
        ],
    },
)
