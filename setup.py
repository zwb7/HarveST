from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# with open("requirements.txt", "r", encoding="utf-8") as fh:
#     requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="harvest",
    version="1.0.0",
    author="zwb7",
    author_email="",
    description="HarveST: A Graph Neural Network for Spatial Transcriptomics Clustering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zwb7/HarveST.git",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    # install_requires=requirements,
    extras_require={
        "dev": ["pytest", "black", "flake8"],
    },
)
