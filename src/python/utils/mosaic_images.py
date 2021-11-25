import rasterio as rio
from pathlib import Path
from rasterio.merge import merge
import geopandas as gpd


def write_image(
    out_fp, img_arr, profile, band_names=["blue", "green", "red", "nir", "swir"]
):
    """Write array to tif file using specified dir and rasterio.profile.

    Args:
        out_fp (pathlib.PosixPath): Path to the output image file to be written.
        img_arr (numpy.Array): Numpy array in rasterio module order.
        profile (dict): Rasterio profile metadata for the image to be written.
        band_names (list): Names of each image band in order. Written in the output geotiff. Defaults to ["blue", "green", "red", "nir", "swir"].
    """
    with rio.open(out_fp, "w", **profile) as f:
        f.descriptions = tuple(band_names)
        f.write(img_arr)
    print(f"image {out_fp} written successfully.")


def boundary_to_mosaic_window(boundary_gdf, crs):
    """Get tuple of xy coordinates from input geodataframe.

    Args:
        boundary_gdf (geopandas.GeoDataFrame): Geodataframe from which full extent coords will be extracted.
        crs (str): The coordinate reference system for the window in EPSG:<code> format.
    Returns:
        tuple: Coordinates for window in minx,miny, maxx, maxy order.
    """
    boundary_gdf = boundary_gdf.to_crs(crs)
    return tuple(boundary_gdf.total_bounds)


def mosaic_image(file_list):
    """Mosaic a list of images returning array and updated rasterio profile for the new image.

    Args:
        file_list (list): List of full paths to the images to be merged.

    Returns:
        Numpy.array, dict: Array of the resulting merged image, plus its updated rasterio metadata profile.
    """
    # To Do: Consider if method=max is the best way to handle overlaps where cloud differs etc
    merged_arr, trans = merge(file_list, method="max")
    with rio.open(file_list[0]) as f:
        prof = f.profile
    prof.update(transform=trans, width=merged_arr.shape[2], height=merged_arr.shape[1])
    return merged_arr, prof


def main(
    input_img_files,
    output_name,
    band_names=["blue", "green", "red", "nir", "swir"],
):
    """Runs main process to create mosaic image from input files.

    Args:
        input_images (list): List of image files to be mosaiced found in data/s2_tiles folder.
        out_name (str): Name for the resulting merged image file.
        scale_value (int, optional): Divide the array by this value so scaled. Defaults to 10000.
    """
    # Get directories
    script_dir = Path(__file__).resolve()
    data_dir = script_dir.parents[2] / "data"
    imgs_dir = script_dir.parents[2] / "data" / "image_tiles"
    # Get full paths to input images
    input_files = [str(imgs_dir / i) for i in input_img_files]
    # Read the boundary layer and convert extent to coord list
    with rio.open(input_files[0]) as f:
        crs = f.crs.to_string()
    # Make the mosaic and scale
    merged_arr, prof = mosaic_image(input_files)
    # Set correct metadata and outputs and write out
    merged_arr = merged_arr.astype("uint16")
    prof.update(dtype="uint16", count=5)
    out_fp = data_dir / output_name
    write_image(out_fp, merged_arr, prof, band_names)


def mk_arg_pars():
    """Create a comand line arg parse.
    Returns:
        dict: Argparse argument dictionary.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Mosaic a list of images into one output tif."
    )

    parser.add_argument(
        "-i",
        "--input_img-files",
        nargs="+",
        default=[
            "sentinel2-test1.tif.tif",
            "sentinel2-test2.tif.tif",
            "sentinel2-test3.tif.tif",
        ],
        help="Specify input tif files to be mosaiced separated by a space. Must be in data/image_tiles dir. Default sentinel2-test1.tif sentinel2-test2.tif sentinel2-test3.tif",
    )
    parser.add_argument(
        "-o",
        "--output-name",
        default="sentinel2-test-all.tif",
        help="Specify the output mosaic image file name ending with .tif. Default sentinel2-test-all.tif",
    )
    parser.add_argument(
        "-n",
        "--band-names",
        default=["blue", "green", "red", "nir", "swir"],
        help="Specify the names of the bands that will be created in the output image. Defaults to ['blue', 'green', 'red', 'nir', 'swir'].",
    )
    args_pars = parser.parse_args()
    return vars(args_pars)


if __name__ == "__main__":
    run_dict = mk_arg_pars()
    main(**run_dict)
