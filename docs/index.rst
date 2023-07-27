.. gspatial-plot documentation master file, created by
   sphinx-quickstart on Wed Feb  8 18:48:05 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

  
Welcome to gspatial-tools's documentation!
=========================================
gspatial_tools is a library of utility functions written on top of geopandas, xarray, rasterio and rioxarray. It makes a lot of things we want to do easier
gspatial-plot is licensed under MIT License.

Features
==========

1. Generate Lat Lon grid at specific resolution
2. Filter points based on radius
3. Bulk Clip and Save Raster Files
4. Reproject Raster Files
5. Stitch Multiple Raster Files in same CRS
6. Read Raster File directly to GeoDataFrame
7. Convert Xarray to GeoDataFrame
8. Clip each geometry by rectangle/bbox
9. Sample points inside polygons
10. Sample points inside bbox
11. Sample points data from raster without loading the entire dataset
12. Get K nearest points along with distance and indices for each point in a GeoDataFrame
13. Move and Scale a polygon/shape from a GeoDataFrame  

Installing
============

`pip install gspatial-tools`

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   modules



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
