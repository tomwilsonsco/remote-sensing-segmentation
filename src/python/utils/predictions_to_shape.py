from pandas.core.base import DataError
import rasterio as rio
from rasterio.features import shapes
import geopandas as gpd
from pathlib import Path
from skimage.morphology import binary_erosion, binary_dilation
import numpy as np
import re


def classify_pred_arr(prob_arr, prediction_threshold=0.5):
    """Create binary array based on a threshold
    Args:
        prob_arr (numpy.array): Array of prediction values to be thresholded
        prediction_threshold (float, optional): Values >= this set to 1 otherwise 0. Defaults to 0.5.
    Returns:
        np.array: uint8 array.
    """
    return (prob_arr >= float(prediction_threshold)).astype("uint8")


def clean_class_img(class_arr, erosion_size=3, dilation_size=3):
    """Removes 'speckle' from binary classification thrugh erosion & dilation.

    Args:
        class_arr (numpy.array): Array of binary values.
    Returns:
        numpy.array: Array of binary values cleaned by erosion then dilation.
    """
    clean_arr = binary_erosion(class_arr, np.ones([erosion_size, erosion_size]))
    clean_arr = binary_dilation(clean_arr, np.ones([dilation_size, dilation_size]))
    return clean_arr.astype("uint8")


def area_filter_gdf(pred_gdf, min_area=2000):
    """Clean polygon geodataframe so outputs greater than or equal to minimum threshold
    Args:
        pred_gdf (geopandas.GeoDataFrame): Geodataframe with polygon geometries in geometry column
        min_area (int): Area in metres squared below which polygons will be dropped. Defaults to 5000.
    Returns:
        geopandas.GeoDataFrame: Geodataframe of just polygons above minimum size
    """
    if pred_gdf.crs.to_string() == "EPSG:4326":
        print("prediction image not in projected crs, cannot filter by area")
        return pred_gdf
    else:
        pred_gdf["area"] = pred_gdf["geometry"].area
        pred_gdf = pred_gdf[pred_gdf["area"] >= min_area]
        return pred_gdf


def dissolve_contiguous(input_gdf):
    """Dissolve geometries into one polygon then explode to ensure no multiparts.
    Args:
        input_gdf (geopandas.GeoDataFrame): Geodataframe with polygon geometry column.
    Returns:
        geopandas.GeoDataFrame: Polygon geodataframe with cleaned internal boundaries.
    """
    input_gdf["diss"] = 1
    input_gdf = input_gdf.dissolve(by="diss", as_index=False)
    input_gdf = input_gdf.explode(index_parts=True)
    return input_gdf


def classified_img_to_shp(class_arr, img_meta):
    """Convert input binary array to polygon geodataframe where 1 pixels converted to polygon.
    Args:
        class_img (numpy.array): 2D binary numpy array.
        img_meta (dict): Rasterio.profile of image metadata for binary array.
    Returns:
        geopandas.GeoDataFrame: Geodataframe of polygons.
    """
    rshapes = (
        {"properties": {"uniqueid": i}, "geometry": s}
        for i, (s, v) in enumerate(
            shapes(class_arr, mask=class_arr, transform=img_meta["transform"])
        )
    )
    geometry = list(rshapes)
    if len(geometry) > 0:
        polygons = gpd.GeoDataFrame.from_features(
            geometry, crs=img_meta["crs"].to_string()
        )
        polygons = dissolve_contiguous(polygons)
        return polygons
    else:
        print("no classified features to be written to shape")
        return None


def remove_mask_adjacent(check_gdf, mask_dir, mask_images):
    """Remove predictions adjacent to masked out areas, e.g. cloud.

    Args:
        check_gdf (geopandas.GeoDataFrame): Polygons to check for adjacency to mask image.
        mask_dir (PosixPath): Where to find the mask images to compare.
        mask_images (list): List of mask image file names to check found within mask_dir.

    Returns:
        geopandas.GeoDataFrame: Geodataframe polygons remaining after removing those adjacent to mask areas.
    """
    from image_binary_mask import invert_arr

    if mask_images is None:
        return check_gdf
    else:
        with rio.open(mask_dir / mask_images[0]) as f:
            y = f.height
            x = f.width
            prof = f.meta
        mask_arr = np.zeros([y, x])
        for mask in mask_images:
            with rio.open(mask_dir / mask) as f:
                current_arr = f.read(1)
                current_arr = invert_arr(current_arr)
                mask_arr += current_arr
        mask_arr = mask_arr.astype("uint8")
        mask_gdf = classified_img_to_shp(mask_arr, prof)
        mask_gdf["geometry"] = mask_gdf["geometry"].buffer(100)
        return check_gdf[~check_gdf.intersects(mask_gdf.unary_union)]


def main_subf(
    input_prediction_image,
    mask_images,
    prediction_threshold=0.5,
    no_filter=False,
    min_area=5000,
):
    """Write polygon shapefile from prediction tif image file.
    Args:
        input_prediction_image (str): Name of tif image file residing in Data dir.
        output_shapefile (str): Name of output shapefile to be written to Data dir.
        prediction_threshold (float, optional): Pixel values >= to threshold will be written to polygon. Defaults to 0.5.
        min_area (int, optional): Polygon geometries areas >= to min_area allowed in output shapefile. Defaults to 5000.
    """
    script_path = Path(__file__).resolve()
    img_dir = script_path.parents[2] / "data" / "prediction_images"
    mask_dir = script_path.parents[2] / "data" / "mask_images"
    img_yrs = re.findall(r"[2][0][1-2][p0-9]", input_prediction_image)
    img_type = re.findall("sentinel2|planet_4m|planet_05m", input_prediction_image)[0]
    if len(img_yrs) < 1 or len(img_yrs) > 2:
        raise DataError(
            "Input image file name must contain 1 or 2 4-digit years 2010 - 2029"
        )
    elif len(img_yrs) == 2 and "minus" in input_prediction_image:
        if int(img_yrs[0]) < int(img_yrs[1]):
            name_fix = f"pred_deforestation_{img_type}_{img_yrs[0]}_{img_yrs[1]}.shp"
        elif int(img_yrs[0]) > int(img_yrs[1]):
            name_fix = f"pred_afforestation_{img_type}_{img_yrs[1]}_{img_yrs[0]}.shp"
        else:
            raise DataError("Check image years in input tif file name and try again.")
    else:
        name_fix = f"treecover_{img_type}_{max(img_yrs)}.shp"
    with rio.open(img_dir / input_prediction_image) as f:
        img_arr = f.read(1)
        prof = f.profile
    class_img = classify_pred_arr(img_arr, prediction_threshold=prediction_threshold)
    if not no_filter:
        print("cleaning classified image...")
        class_img = clean_class_img(class_img)
    print("creating prediction shapes...")
    pred_gdf = classified_img_to_shp(class_img, prof)
    if pred_gdf is not None:
        shp_dir = script_path.parents[2] / "data" / "prediction_shapes"
        Path.mkdir(shp_dir, exist_ok=True)
        print("checking for masked areas...")
        pred_gdf = remove_mask_adjacent(pred_gdf, mask_dir, mask_images)
        pred_gdf = area_filter_gdf(pred_gdf, min_area)
        out_fp = shp_dir / name_fix
        pred_gdf.to_file(out_fp, index=False)
        print(f"{len(pred_gdf)} result polygons written result to {out_fp}")


def mk_arg_pars():
    """Create a comand line arg parse.
    Returns:
        dict: Argparse argument dictionary.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert prediction image to shapefile polygons"
    )
    parser.add_argument(
        "-i",
        "--input-prediction-image",
        default="sentinel2-test-predictions.tif",
        help="Specify .tif image of predictions from which shapes will be extracted. Must be in Data dir. Default sentinel2-test-predictions.tif.",
    )

    parser.add_argument(
        "-t",
        "--prediction-threshold",
        default=0.5,
        help="Specify threshold for positive classification above which polygons will be created. Defaults to 0.5",
    )

    parser.add_argument(
        "-nf",
        "--no-filter",
        action="store_true",
        help="Add this option if do not want to filter the classified image with erosion and dilation filters",
    )

    parser.add_argument(
        "-a",
        "--min-area",
        default=2000,
        help="Specify minimum area allowed for the predictions in sq m. Defaults to 2000 sq m",
    )
    parser.add_argument(
        "-m",
        "--mask-images",
        nargs="*",
        help="Optionally specify one or more mask images. Result polygons adjacent to mask will be removed.",
    )

    args_pars = parser.parse_args()
    return vars(args_pars)


if __name__ == "__main__":
    run_dict = mk_arg_pars()
    main_subf(**run_dict)
