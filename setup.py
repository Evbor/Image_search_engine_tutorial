from distutils.core import setup

setup(
    name="imgsrcheng",
    version="0.1",
    description="Image Search Engine",
    long_description="builds and runs an image search engine",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    url="https://github.com/Evbor/Image_search_engine_tutorial",
    install_requires=[
        'numpy',
        'opencv-python',
        'scipy',
        'tqdm'
    ],
    include_package_data=True,
    author="Evbor",
    author_email='',
    license="MIT",
    packages=["imgsrcheng"]
)
