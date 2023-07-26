import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import json
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.merge import merge
import rioxarray


def generate_points_grid(bounds, grid_space=0.01, crs="4326", clip=False, shape=None):
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
    if clip==True:
        grid = grid.clip(shape)
    return grid


def filter_points_to_radius(grid, points, radius, cap_style=3):
    points_buffered = points.copy()
    points_buffered["geometry"] = points_buffered.buffer(radius, cap_style=cap_style)
    points_buffered.crs = points.crs
    points_buffered = points_buffered[["geometry"]]
    filtered = gpd.sjoin(grid, points_buffered)
    return filtered


def bulk_clip_and_save(data, gdf, names_col=None, path=None):
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


def read_raster_as_gdf(filename, x=None, y=None, name=None, dropna=True):
    try:
        data = rioxarray.open_rasterio(filename)
        try:
            df = data.to_dataframe().reset_index()
        except:
            if name is not None:
                df = data.to_dataframe(name=name).reset_index()
            else:
                df = data.to_dataframe(name="data").reset_index()
        if dropna == True:
            df.dropna(inplace=True)
        try:
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x"], df["y"]))
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


def convert_xarray_to_gdf(data, x=None, y=None, engine=None, dropna=True):
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
