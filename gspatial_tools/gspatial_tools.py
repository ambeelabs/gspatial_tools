import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import json
import rioxarray
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.merge import merge
from shapely.ops import clip_by_rect


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
        _type_: _description_
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
        _type_: _description_
    """
    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].apply(
        lambda x: clip_by_rect(x, xmin, ymin, xmax, ymax)
    )
    return gdf
