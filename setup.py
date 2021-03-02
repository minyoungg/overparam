import setuptools

setuptools.setup(
    name="overparam",
    version="0.0.1",
    author="Minyoung Huh",
    author_email="minhuh@mit.edu",
    description=f"automatic linear over-parameterization PyTorch layers for " +\
                  "`The Low-Rank Simplicity Bias in Deep Networks`. " +\
                  "Code has been tested on Python 3.7 and PyTorch 1.7",
    url="git@github.com:minyoungg/overparam-minimal.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
