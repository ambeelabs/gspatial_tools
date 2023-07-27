import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import json
import rioxarray


def generate_points_grid(bounds, grid_space=0.01, crs="4326", clip=False, shape=None):
    """Generates a grid of point data for given bounds. The data can be clipped by passing geometries for clipping supported by Geopandas

    Args:
        bounds (list): Bounding box for the grid
        grid_space (float, optional): Resolution of the grid. Grid space needs to be chosen according to crs. Defaults to 0.01.
        crs (str, optional): Coordinate Reference System for the grid. Defaults to "4326".
        clip (bool, optional): Clips by shape parameter if True. Defaults to False.
        shape (Polygon/GeoSeries/GeoDataframe/Anything supported by Geopandas Clip, optional): Shape to clip. Should be in same CRS as grid. Defaults to None.

    Returns:
        GeoDataFrame: GeoDataFrame containing points
    """
    grid_lon = np.arange(bounds[0], bounds[2], grid_space)
    grid_lat = np.arange(bounds[1], bounds[3], grid_space)
    all_lats = np.meshgrid(grid_lon, grid_lat)[1].ravel()
    all_lons = np.meshgrid(grid_lon, grid_lat)[0].ravel()
    del grid_lat, grid_lon

    pairs = list(zip(all_lats, all_lons))
    del all_lats, all_lons

    grid = pd.DataFrame(pairs, columns=["lat", "lon"])
    grid = gpd.GeoDataFrame(
        grid, geometry=gpd.points_from_xy(grid["lon"], grid["lat"]), crs=crs
    )
    if clip == True:
        grid = grid.clip(shape)
    return grid


def filter_points_to_radius(grid, points, radius, cap_style=3):
    """Filters grid to a certain radius taking the point passed as center.

    Args:
        grid (GeoDataFrame): Grid of points
        points (GeoSeries/GeoDataFrame): Set of points need to filter the grid to a certain radius. Should be in same CRS as grid.
        radius (float): Radius to filter points. Should be in same CRS as grid and points
        cap_style (int, optional): cap_style parameter to buffer function. Defaults to 3 which is a square.

    Returns:
        GeoDataFrame: filtered GeoDataFrame
    """
    points_buffered = points.copy()
    points_buffered["geometry"] = points_buffered.buffer(radius, cap_style=cap_style)
    points_buffered.crs = points.crs
    points_buffered = points_buffered[["geometry"]]
    filtered = gpd.sjoin(grid, points_buffered)
    return filtered


def bulk_clip_and_save_raster(data, gdf, names_col=None, path=None):
    """Utility function to clip and save rasters in bulk using a GeoDataFrame.

    Args:
        data (raster/ndarray): Data read from rasterio
        gdf (GeoDataFrame): Shapes to clip the raster
        names_col (str, optional): Column name to name to file saved. Defaults to None.
        path (str, optional): Path to save clipped files. Defaults to None.
    """
    from rasterio.mask import mask

    gdf = gdf.copy()
    gdf = gdf.to_crs(data.crs)
    gdf_json = json.loads(gdf.to_json())
    coords = [
        [gdf_json["features"][i]["geometry"]] for i in range(len(gdf_json["features"]))
    ]
    if names_col is None:
        names = [str(i) for i in range(len(gdf_json["features"]))]
    else:
        names = [
            gdf_json["features"][i]["properties"][names_col]
            for i in range(len(gdf_json["features"]))
        ]
    for coord, name in zip(coords, names):
        out_image, out_transform = mask(data, shapes=coord, crop=True)
        out_meta = data.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )
        if path is None:
            filename = name + ".tif"
        else:
            filename = path + name + ".tif"
        with rasterio.open(filename, "w", **out_meta, compress="lzw") as dest:
            dest.write(out_image)


def reproject_rasterio(data, dst_crs, output_file):
    """Utility function to reproject and save raster in rasterio

    Args:
        data (raster/ndarray): Data read from rasterio
        dst_crs (str): Target crs supported by rasterio
        output_file (str): Output file name
    """
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    transform, width, height = calculate_default_transform(
        data.crs, dst_crs, data.width, data.height, *data.bounds
    )
    kwargs = data.meta.copy()
    kwargs.update(
        {"crs": dst_crs, "transform": transform, "width": width, "height": height}
    )

    with rasterio.open(output_file, "w", compress="lzw", **kwargs) as dst:
        for i in range(1, data.count + 1):
            reproject(
                source=rasterio.band(data, i),
                destination=rasterio.band(dst, i),
                src_transform=data.transform,
                src_crs=data.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
            )


def stitch_rasters(file_list, out_path):
    """Merges a list of raster files into a single file. (All need to be in same CRS)

    Args:
        file_list (list): List of filenames
        out_path (str): Output file path/name
    """
    from rasterio.merge import merge

    to_stitch = []
    for file in file_list:
        raster = rasterio.open(file)
        to_stitch.append(raster)
    out_image, out_transform = merge(to_stitch)
    out_meta = raster.meta.copy()
    out_meta.update(
        {
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        }
    )
    with rasterio.open(out_path, "w", **out_meta, compress="lzw") as dest:
        dest.write(out_image)


def read_raster_as_gdf(filename, x="x", y="y", name="data", dropna=True):
    """Read raster file as GeoDataFrame

    Args:
        filename (str): Name of raster file
        x (str, optional): column/field name for x coordinate. Defaults to "x".
        y (str, optional): column/field name for y coordinate. Defaults to "y".
        name (str, optional): Name for. Defaults to "data".
        dropna (bool, optional): Drops null value from raster. Defaults to True.

    Raises:
        e: Any exception raised during the process

    Returns:
        gdf: GeoDataFrame
    """
    try:
        data = rioxarray.open_rasterio(filename)
        try:
            df = data.to_dataframe().reset_index()
        except:
            df = data.to_dataframe(name=name).reset_index()
        if dropna == True:
            df.dropna(inplace=True)
        try:
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[x], df[y]))
        except Exception as e:
            print(
                "Failed to convert to gdf, Please specify correct x,y values or try to open in xarray and convert"
            )
            print("Indices")
            print(data.indexes)
            print("Coordinates")
            print(data.coords)
            raise e
        return gdf
    except Exception as e:
        print("Failed to open raster")
        print("If your raster is in hdf format, please install gdal from condaforge")
        raise (e)


def convert_xarray_to_gdf(data, x="x", y="y", dropna=True):
    """Converts xarray to gdf

    Args:
        data (xarray Dataset/DataArray): xarray data to be converted
        x (str, optional): column/field name for x coordinate. Defaults to "x".
        y (str, optional): column/field name for y coordinate. Defaults to "y".
        dropna (bool, optional): Drops null value from raster. Defaults to True.

    Raises:
        e: Any exception raised during the process

    Returns:
        gdf: GeoDataFrame
    """
    df = data.to_dataframe().reset_index()
    if dropna == True:
        df.dropna(inplace=True)
    try:
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[x], df[y]))
    except Exception as e:
        print(
            "Failed to convert to gdf, Please specify correct x,y values or try to open in xarray and convert"
        )
        print("Indices")
        print(data.indexes)
        print("Coordinates")
        print(data.coords)
        raise e
    return gdf


def clip_each_geom_by_rect(gdf, xmin, ymin, xmax, ymax):
    """Clips each geometry in a Dataframe by rectangle and returns the result

    Args:
        gdf (GeoDataFrame): Input GeoDataFrame
        xmin (float): Minimum x bounds
        ymin (float): Minimum y bounds
        xmax (float): Maximum x bounds
        ymax (float): Maximum y bounds

    Returns:
        gdf: GeoDataFrame
    """
    from shapely.ops import clip_by_rect

    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].apply(
        lambda x: clip_by_rect(x, xmin, ymin, xmax, ymax)
    )
    return gdf


def sample_points_from_polygons(gdf, n: int, crs=None):
    """Samples n points inside set of polygons

    Args:
        gdf (GeoDataFrame/GeoSeries): GeoDataFrame/GeoSeries containing polygons
        n (int): number of samples
        crs (str): CRS for the points, If nothing is passed, CRS of gdf is set. Defaults to None

    Returns:
        points: A Geoseries containing sampled points
    """
    points = []
    while len(points) < n:
        x = np.random.uniform(gdf.total_bounds[0], gdf.total_bounds[2], n)
        y = np.random.uniform(gdf.total_bounds[1], gdf.total_bounds[3], n)
        points_series = gpd.GeoSeries(gpd.points_from_xy(x, y))
        points_series = points_series[points_series.within(gdf.unary_union)]
        points_series = points_series.drop_duplicates()
        points.extend(list(points_series))
        points = list(set(points))
    points = gpd.GeoSeries(points)
    points = points.sample(n).reset_index(drop=True).drop_duplicates()
    if crs == None:
        try:
            points.crs = gdf.crs
        except:
            print("Could not set CRS")
    else:
        points.crs = crs
    return points


def sample_points_from_bbox(bounds, n: int, crs=None):
    """Samples n points inside a bounding box

    Args:
        bounds (list): bounding box/total_bounds of a GeoDataFrame/GeoSeries/Polygon
        n (int): number of samples

    Returns:
       points: A Geoseries containing sampled points
    """
    x = np.random.uniform(bounds[0], bounds[2], n)
    y = np.random.uniform(bounds[1], bounds[3], n)
    points = gpd.GeoSeries(gpd.points_from_xy(x, y))
    if crs is not None:
        points.crs = crs
    return points


def sample_data_from_raster(data, points, col_name="data", n_bands=1):
    """Samples data directly from a raster file

    Args:
        data (rasterio.io.DatasetReader): Raster file to sample data from
        points (GeoDataFrame/GeoSeries): A GeoDataFrame/GeoSeries containing Point geometries.
        col_name (str, optional): Name of the column containing sampled_data. Defaults to "data".
        n_bands (int, optional): Number of bands. Defaults to 1.
    Raises:
        e: Any exception due to crs issues

    Returns:
        points: GeoDataFrame
    """
    try:
        points = points.to_crs(data.crs)
    except Exception as e:
        print("Could not convert to data's CRS, ensure that CRS is set for points")
        raise e
    try:
        coords = [(x, y) for x, y in zip(points["geometry"].x, points["geometry"].y)]
    except:
        points = gpd.GeoDataFrame(geometry=points)
        coords = [(x, y) for x, y in zip(points["geometry"].x, points["geometry"].y)]
    if n_bands == 1:
        points[col_name] = [float(x) for x in data.sample(coords)]
    else:
        points[col_name] = [x for x in data.sample(coords)]
    return points


def nearest_points(
    left_gdf,
    right_gdf,
    k=3,
    leaf_size=15,
    distance_unit="radians",
    return_indices=False,
):
    """Returns k nearest points to left GeoDataFrame

    Args:
        left_gdf (GeoDataFrame): Dataframe for which nearest points need to be calculated
        right_gdf (GeoDataFrame): Dataframe which contains the set of points from which nearest points will be calculated
        k (int, optional): Number of nearesr points. Defaults to 3.
        leaf_size (int, optional): Leafsize parameter for ball. Defaults to 15.
        distance_unit (str, optional): Unit for distance. If "meters" or "kilometers" is passed, It'll approximate based on earth's radius. Defaults to "radians".
        return_indices (bool, optional): If True, indices will be returned. Defaults to False.

    Returns:
        result: GeoDataFrame containing results
    """
    from sklearn.neighbors import BallTree

    if left_gdf.crs != right_gdf.crs:
        print("CRS of both GeoDataFrames should be the same")
        return None
    crs = left_gdf.crs
    left_gdf = left_gdf.to_crs("4326")
    right_gdf = right_gdf.to_crs("4326")

    left_gdf_radians = np.array(
        left_gdf["geometry"]
        .apply(lambda coords: (np.deg2rad(coords.y), np.deg2rad(coords.x)))
        .to_list()
    )
    right_gdf_radians = np.array(
        right_gdf["geometry"]
        .apply(lambda coords: (np.deg2rad(coords.y), np.deg2rad(coords.x)))
        .to_list()
    )

    tree = BallTree(right_gdf_radians, leaf_size=leaf_size, metric="haversine")
    distances, indices = tree.query(left_gdf_radians, k=k)
    del left_gdf_radians, right_gdf_radians

    left_gdf = left_gdf.to_crs(crs)
    right_gdf = right_gdf.to_crs(crs)

    indices_map = right_gdf["geometry"].to_dict()
    indices_df = pd.DataFrame(indices)
    geom_df = indices_df.applymap(indices_map.get)
    del indices_map, right_gdf, indices

    distances_df = pd.DataFrame(distances)
    geom_df.columns = ["nearest_geom_" + str(x) for x in geom_df.columns]
    distances_df.columns = ["distance_" + str(x) for x in distances_df.columns]
    if distance_unit == "meters":
        distances_df = distances_df * 6371000
    if distance_unit == "kilometers":
        distances_df = distances_df * 6371
    result = left_gdf.copy()
    del left_gdf, distances

    result[geom_df.columns] = geom_df
    result[distances_df.columns] = distances_df
    if return_indices == True:
        indices_df.columns = ["index_" + str(x) for x in indices_df.columns]
        result[indices_df.columns] = indices_df
    return result


def move_and_scale_shape(
    gdf, identifier_col, identifier_value, scale_factor, x_distance, y_distance
):
    """Move and scale a specific polygon in a GeoDataFrame

    Args:
        gdf (GeoDataFrame): GeoDataFrame to be modified
        identifier_col (str): Column name to identify polygon/shape
        identifier_value (str): Value corresponding to the column name
        scale_factor (float): Scale factor to resize the polygon
        x_distance (float): Offset in x coordinate
        y_distance (float): Offset in x coordinate

    Returns:
        modified_gdf (GeoDataFrame): Returns the modfied GeoDataFrame
    """
    from shapely.affinity import scale, translate

    shape_polygon = gdf.loc[gdf[identifier_col] == identifier_value, "geometry"].iloc[0]
    shape_scaled = scale(
        shape_polygon, xfact=scale_factor, yfact=scale_factor, origin="centroid"
    )
    shape_moved = translate(shape_scaled, xoff=x_distance, yoff=y_distance)
    modified_gdf = gdf.copy()
    modified_gdf.loc[
        modified_gdf[identifier_col] == identifier_value, "geometry"
    ] = shape_moved
    return modified_gdf
