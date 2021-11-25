import rasterio as rio
from pathlib import Path
import re
import json
import numpy as np
from rasterio.enums import Compression
from predictions_to_shape import clean_class_img


def get_config(input_image, no_data_config):
    """Read config json file for valid pixel value ranges per band and image type.

    Args:
        input_image (str): Name of the image file - must contain image type e.g. sentinel2, planet_4m.
        no_data_config (dict): Config dictionary as found in json file in input dir.

    Returns:
        str, dict: Name of image file type, dictionary of valid pixel ranges for that image type.
    """
    image_type = re.findall("sentinel2|planet_4m", input_image)[0]
    return image_type, no_data_config[image_type]


def calc_no_data(img_arr, image_type, valid_dict):
    """Based on config dictionary of valid pixel value ranges reclassify image into 1 band binary image.

    Args:
        img_arr (numpy.array): Numpy array of image file in rasterio order (bands, y, x).
        image_type (str): Image type. One of sentinel or planet_4m.
        valid_dict (dict): Dictionary of valid values for rgb, nir or swir image bands.

    Returns:
        numpy.array: 1-band binary array in image dims shape, 1 being valid and 0 invalid values based on ranges.
    """
    # Scale array to 0-1
    img_arr = img_arr / 10000
    # Get indices and ranges for bands
    red_idx = valid_dict["bands"].index("r")
    green_idx = valid_dict["bands"].index("g")
    blue_idx = valid_dict["bands"].index("b")
    if "sentinel" in image_type:
        ir_idx = valid_dict["bands"].index("swir")
        rgb_min, ir_min, rgb_max, ir_max = (
            valid_dict["rgb_min"],
            valid_dict["swir_min"],
            valid_dict["rgb_max"],
            valid_dict["swir_max"],
        )
    else:
        ir_idx = valid_dict["bands"].index("nir")
        rgb_min, ir_min, rgb_max, ir_max = (
            valid_dict["rgb_min"],
            valid_dict["nir_min"],
            valid_dict["rgb_max"],
            valid_dict["nir_max"],
        )
    # Classify ranges to 0,1 binary array
    below_min = (
        (img_arr[red_idx] < rgb_min)
        & (img_arr[green_idx] < rgb_min)
        & (img_arr[blue_idx] < rgb_min)
        & (img_arr[ir_idx] < ir_min)
    )
    above_max = (
        (img_arr[red_idx] > rgb_max)
        & (img_arr[green_idx] > rgb_max)
        & (img_arr[blue_idx] > rgb_max)
        & (img_arr[ir_idx] > ir_max)
    )
    if "b_max" in valid_dict.keys():
        b_max = valid_dict["b_max"]
        above_max = above_max | (img_arr[blue_idx] > b_max)
    return (below_min | above_max).astype("uint8")


def invert_arr(input_arr):
    """Invert binary array using numpy.invert"""
    return np.where(input_arr == 0, 1, 0).astype("uint8")


def main_subf(input_image):
    """Main function to write image mask tif file into mask_images sub dir based on input image and json config file.

    Args:
        input_image (str): Image tif file to process that must exist in data dir.
    """
    script_dir = Path(__file__).resolve()
    data_dir = script_dir.parents[2] / "data"
    input_dir = script_dir.parents[2] / "input"
    mask_dir = data_dir / "mask_images"
    mask_dir.mkdir(exist_ok=True)
    with rio.open(data_dir / input_image) as f:
        prof = f.profile
        img_arr = f.read()
    with open(input_dir / "no_data_ranges.json") as json_file:
        no_data_config = json.load(json_file)
    image_type, valid_dict = get_config(input_image, no_data_config)
    valid_arr = calc_no_data(img_arr, image_type, valid_dict)
    valid_arr = clean_class_img(valid_arr, erosion_size=5, dilation_size=25)
    valid_arr = invert_arr(valid_arr)
    prof.update(count=1, dtype="uint8", compress="lzw")
    out_name = f"{input_image.replace('.tif', '')}_mask.tif"
    with rio.open(mask_dir / out_name, "w", **prof) as f:
        f.write(valid_arr, indexes=1)
    print(f"written {out_name} to {mask_dir}")


def mk_arg_pars():
    """Create a comand line arg parse.
    Returns:
        dict: Argparse argument dictionary.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a binary tif image to mask out non-valid data in input tif image file."
    )
    parser.add_argument(
        "-i",
        "--input-image",
        default="sentinel2-test.tif",
        help="Specify relative path to image file under the data dir. Default sentinel2-test.tif.",
    )

    args_pars = parser.parse_args()
    return vars(args_pars)


if __name__ == "__main__":
    run_dict = mk_arg_pars()
    main_subf(**run_dict)
