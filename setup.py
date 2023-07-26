from setuptools import find_packages, setup
import os

here = os.path.abspath(os.path.dirname(__file__))

long_description = open(os.path.join(here, 'README.md')).read()

setup(
    name="gspatial_tools",
    packages=find_packages(include=["gspatial_tools"]),
    version="0.1.0a",
    author="Ambee",
    license="MIT",
    url='https://github.com/ambeelabs/gspatial_tools/tree/main/gspatial_tools',
    description="Set of utility tools built on top of Geopandas, Xarray, Rasterio and Rioxarray",
    long_description_content_type="text/markdown",
    long_description=long_description,
    install_requires=[
    ],
     classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
    ],
    keywords='geospatial utilities geopandas rasterio rioxarray xarray',
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    test_suite="tests",
    include_package_data=True,
)
