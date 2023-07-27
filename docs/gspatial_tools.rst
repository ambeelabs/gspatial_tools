gspatial\_tools package
======================
The examples mentioned in the documentation is also part of examples jupyter notebook present in the github repo, So feel free to download it and play with it.

.. py:function:: generate_points_grid(bounds, grid_space=0.01, crs="4326", clip=False, shape=None)
    
    Generates a grid of point data for given bounds. The data can be clipped by passing geometries for clipping supported by Geopandas

    :param bounds: Bounding box for the grid
    :type bounds: list
    :param grid_space: Resolution of the grid. Grid space needs to be chosen according to crs. Defaults to 0.01.
    :type grid_space: float
    :param crs: Coordinate Reference System for the grid. Defaults to "4326".
    :type crs: str
    :param clip: Clips by shape parameter if True. Defaults to False.
    :type clip: bool
    :param shape: Shape to clip. Should be in same CRS as grid. Defaults to None.
    :type shape: Polygon/GeoSeries/GeoDataframe/Anything supported by Geopandas Clip
    :returns: GeoDataFrame containing points
    :rtype: GeoDataFrame

**Examples**

.. code:: ipython3

    grid = generate_points_grid(argentina.total_bounds,grid_space=1)

.. code:: ipython3

    grid = generate_points_grid(argentina.total_bounds,grid_space=1, clip=True, shape=argentina)


.. py:function:: filter_points_to_radius(grid, points, radius, cap_style=3)
    
    Filters grid to a certain radius taking the point passed as center.

    :param grid: Grid of points
    :type grid: GeoDataFrame
    :param points: Set of points need to filter the grid to a certain radius. Should be in same CRS as grid.
    :type points: GeoSeries/GeoDataFrame
    :param radius: Radius to filter points. Should be in same CRS as grid and points
    :type radius: float
    :param cap_style: cap_style parameter to buffer function. Defaults to 3 which is a square.
    :type cap_style: int
    :returns: filtered GeoDataFrame
    :rtype: GeoDataFrame
    
**Examples**

.. code:: ipython3

    filtered = filter_points_to_radius(grid, point, 5)



.. py:function:: bulk_clip_and_save_raster(data, gdf, names_col=None, path=None)
 
    Utility function to clip and save rasters in bulk using a GeoDataFrame.

    :param data: Data read from rasterio
    :type data: raster/ndarray
    :param gdf: Shapes to clip the raster
    :type gdf: GeoDataFrame
    :param names_col: Column name to name to file saved. Defaults to None.
    :type names_col: str
    :param path: Path to save clipped files. Defaults to None.
    :type path: str

**Examples**
bulk_clip_and_save_raster(data, world, names_col="name")


.. py:function:: reproject_rasterio(data, dst_crs, output_file)
 
    Utility function to reproject and save raster in rasterio

    :param data: Data read from rasterio
    :type data: raster/ndarray
    :param dst_crs: Target crs supported by rasterio
    :type dst_crs: str
    :param output_file: Output file name
    :type output_file: str

**Examples**

.. code:: ipython3

    japan = rasterio.open("Japan.tif")
    reproject_rasterio(japan,"EPSG:4326","Japan_WGS84.tif")

.. py:function:: stitch_rasters(file_list, out_path)
    
    Merges a list of raster files into a single file. (All need to be in same CRS)

    :param file_list: List of filenames
    :type file_list: list
    :param out_path: Output file path/name
    :type out_path: str
    

**Examples**

.. code:: ipython3

    stitch_rasters(file_list,"stitched.tif")


.. py:function:: read_raster_as_gdf(filename, x="x", y="y", name="data", dropna=True)
 
    Read raster file as GeoDataFrame

    :param filename: Name of raster file
    :type filename: str
    :param x: column/field name for x coordinate. Defaults to "x".
    :type x: str
    :param y: column/field name for y coordinate. Defaults to "y".
    :type y: str
    :param name: Name for. Defaults to "data".
    :type name: str
    :param dropna: Drops null value from raster. Defaults to True.
    :type dropna: bool
    :returns: GeoDataFrame
    :rtype: gdf
    :raises e: Any exception raised during the process

**Examples**

.. code:: ipython3

    gdf = read_raster_as_gdf("Japan.tif")


.. py:function:: convert_xarray_to_gdf(data, x="x", y="y", dropna=True)
    
    Converts xarray to gdf

    :param data: xarray data to be converted
    :type data: xarray Dataset/DataArray
    :param x: column/field name for x coordinate. Defaults to "x".
    :type x: str
    :param y: column/field name for y coordinate. Defaults to "y".
    :type y: str
    :param dropna: Drops null value from raster. Defaults to True.
    :type dropna: bool
    :returns: GeoDataFrame
    :rtype: gdf
    :raises e: Any exception raised during the process

**Examples**

.. code:: ipython3

    gdf = convert_xarray_to_gdf(nc, x="lon", y="lat")

.. py:function:: clip_each_geom_by_rect(gdf, xmin, ymin, xmax, ymax)

    Clips each geometry in a Dataframe by rectangle and returns the result

    :param gdf: Input GeoDataFrame
    :type gdf: GeoDataFrame
    :param xmin: Minimum x bounds
    :type xmin: float
    :param ymin: Minimum y bounds
    :type ymin: float
    :param xmax: Maximum x bounds
    :type xmax: float
    :param ymax: Maximum y bounds
    :type ymax: float
    :returns: GeoDataFrame
    :rtype: GeoDataFrame

**Examples**

.. code:: ipython3

    us = clip_each_geom_by_rect(us,-180,-90,-69,83)

.. py:function:: sample_points_from_polygons(gdf, n: int, crs=None)
    
    Samples n points inside set of polygons

    :param gdf: GeoDataFrame/GeoSeries containing polygons
    :type gdf: GeoDataFrame/GeoSeries
    :param n: number of samples
    :type n: int
    :param crs: CRS for the points, If nothing is passed, CRS of gdf is set. Defaults to None
    :type crs: str
    :param n: int: 
    :returns: A Geoseries containing sampled points
    :rtype: points

**Examples**

.. code:: ipython3

    points = sample_points_from_polygons(argentina,200)


.. py:function:: sample_points_from_bbox(bounds, n: int, crs=None)

    Samples n points inside a bounding box

    :param bounds: bounding box/total_bounds of a GeoDataFrame/GeoSeries/Polygon
    :type bounds: list
    :param n: number of samples
    :type n: int
    :param n: int: 
    :param crs:  (Default value = None)
    :returns: A Geoseries containing sampled points
    :rtype: points
    
**Examples**

.. code:: ipython3

    points = sample_points_from_bbox(world.total_bounds,20, crs="4326")


.. py:function:: sample_data_from_raster(data, points, col_name="data", n_bands=1)

    Samples data directly from a raster file

    :param data: Raster file to sample data from
    :type data: rasterio.io.DatasetReader
    :param points: A GeoDataFrame/GeoSeries containing Point geometries.
    :type points: GeoDataFrame/GeoSeries
    :param col_name: Name of the column containing sampled_data. Defaults to "data".
    :type col_name: str
    :param n_bands:  (Default value = 1)
    :returns: points
    :rtype: GeoDataFrame
    :raises e: Any exception due to crs issues

**Examples**

.. code:: ipython3

    sample_points = sample_data_from_raster(argentina_raster, points, col_name="pop")


.. py:function:: nearest_points(left_gdf,right_gdf,k=3,leaf_size=15,distance_unit="radians",return_indices=False)

    Returns k nearest points to left GeoDataFrame

    :param left_gdf: GeoDataFrame
    :param right_gdf: GeoDataFrame
    :param k: int (Default value = 3)
    :param leaf_size: int (Default value = 15)
    :param distance_unit: str (Default value = "radians")
    :param return_indices: bool (Default value = False)
    :returns: result: GeoDataFrame containing results

**Examples**

.. code:: ipython3

    nearest_points(points,points, distance_unit="kilometers", return_indices=True)


.. py:function:: move_and_scale_shape(gdf,identifier_col,identifier_value,scale_factor,x_distance,y_distance)

    Move and scale a specific polygon in a GeoDataFrame

    :param gdf: GeoDataFrame to be modified
    :type gdf: GeoDataFrame
    :param identifier_col: Column name to identify polygon/shape
    :type identifier_col: str
    :param identifier_value: Value corresponding to the column name
    :type identifier_value: str
    :param scale_factor: Scale factor to resize the polygon
    :type scale_factor: float
    :param x_distance: Offset in x coordinate
    :type x_distance: float
    :param y_distance: Offset in x coordinate
    :type y_distance: float
    :returns: modified_gdf-> Returns the modfied GeoDataFrame
    :rtype: GeoDataFrame

**Examples**

.. code:: ipython3

    states = move_and_scale_shape(states,"NAME","Alaska",0.5,30,-40)
    states = move_and_scale_shape(states,"NAME","Hawaii",1,70,0)