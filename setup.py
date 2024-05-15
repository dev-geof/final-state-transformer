import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="final-state-transformer",
    version="0.1.0",
    author="Geoffrey Gilles",
    description="Machine Learning development toolkit built upon Transformer encoder network architectures and tailored for the realm of high-energy physics and particle event analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dev-geof/final-state-transformer",
    packages=setuptools.find_packages(),
    install_requires=[
        "h5py>=3.8.0",
        "matplotlib>=3.5.3",
        "numpy>=1.24.2",
        "puma>=0.0.0rc1",
        "puma_hep>=0.2.2",
        "PyYAML>=6.0",
        "scikit_learn>=1.2.2",
        "tensorflow>=2.11.0",
        "termcolor>=1.1.0",
        "tqdm>=4.62.3",
        "pydot>=1.4.2",
        "graphviz>=0.20.1",
        "tf2onnx>=1.12.0",
        "cuda-python>=12.4.0",
        "scikit-learn>=1.1.2",
    ],
    entry_points={
        "console_scripts": [
            "fst-preparation=src.preparation:main",
            "fst-training=src.training:main",
            "fst-validation=src.validation:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.9.13",
)
