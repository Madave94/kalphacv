import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="iaadet",
    version="1.0",
    author="Henrik Leisdon, David Tschirschwitz",
    author_email="david.tschirschwitz@uni-weimar.de",
    description="Package to calculate inter annotator agreement based on krippendorff's alpha for computer vision",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy',
                      'scipy',
                      'pytest',
                      'krippendorff',
                      'matplotlib',
                      'scikit-image',
                      'imagecodecs',
                      'Pillow',],
    package_data={'src':['*.json', "*.csv"]},
    include_package_data=True,
)
