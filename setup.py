import setuptools

setuptools.setup(
    name="GeoFormer",
    version="0.0.1",
    author="Liang Ding",
    author_email="liangding86@gmail.com",
    description="Predictive lithological mapping and uncertainty quantification with deep learning",
    long_description="Swin Transformer-Based Predictive Lithological Mapping and Uncertainty Quantification",
    long_description_content_type="text/markdown",
    url="https://github.com/Liang-Ding/geoformer",
    project_urls={
        "Bug Tracker": "",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords=[
        "seismology"
    ],
    # package_dir={"": "geoformer"},
    python_requires='>=3.12.0',
    install_requires=[
        "torch", "numpy", "h5py", "timm"
    ],
    packages=setuptools.find_packages(),
)