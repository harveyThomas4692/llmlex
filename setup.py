import setuptools

with open("readme.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

with open("requirements.txt", "r") as fh:
    REQUIREMENTS = fh.read().splitlines()

setuptools.setup(
    name="LLMSR",
    version="0.0.1",
    author="Thomas R. Harvey",
    author_email="trharvey@mit.edu",
    description="An open source python library for symbolic regression with LLMs",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/harveyThomas4692/LLM_SR",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    install_requires=REQUIREMENTS,
)
