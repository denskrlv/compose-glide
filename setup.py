import os

from setuptools import setup, find_packages


# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name="compose-glide",
    version="0.1.0",
    description="Text-to-image generation using enhanced GLIDE models",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/compose-glide",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Researchers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="glide, text-to-image, deep learning, ai, diffusion models",
    # Optional: Add entry points for command line tools
    entry_points={
        "console_scripts": [
            "compose-glide=compose_glide.cli:main",
        ],
    },
    # Include non-code files
    include_package_data=True,
)
