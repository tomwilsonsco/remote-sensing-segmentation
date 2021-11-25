import rasterio as rio
from rasterio.windows import Window
from pathlib import Path
import numpy as np
from skimage.segmentation import quickshift
import plotly.express as px
import scipy
import random


def get_windows(input_img_fp, window_size=1000):
    """Create list of index ranges, diving input image file into windows of specified size.

    Args:
        input_img_fp (str): Full path to the image (geotiff) to be split into windows.
        window_size (int, optional): Windows will be this number of pixels in x,y dimensions. Defaults to 1000.

    Returns:
        list: List of distinct window to from indices covering full input image extent.
    """
    with rio.open(input_img_fp) as f:
        x_max = f.width
        y_max = f.height
    x_ranges = [x for x in range(0, x_max, window_size)]
    y_ranges = [y for y in range(0, y_max, window_size)]
    window_list = [[x, y, window_size, window_size] for x in x_ranges for y in y_ranges]
    for i, w in enumerate(window_list):
        if w[0] + window_size > x_max:
            w[2] = x_max - w[0]
        if w[1] + window_size > y_max:
            w[3] = y_max - w[1]
    return window_list


def get_min_max_range(arr_1d, std_factor=2):
    """Get min max ranges for array, defined as so many standard deviations +/- from the mean

    Args:
        arr_1d (numpy.array): Numpy array over which values are calculated.
        std_factor (int, optional): The min / max ranges returned are this many std +/- the mean. Defaults to 3.

    Returns:
        double, double: min and max calculated values
    """
    arr_mean = arr_1d.mean()
    arr_std = arr_1d.std()
    min_val = arr_mean - arr_std * std_factor
    max_val = arr_mean + arr_std * std_factor
    return min_val, max_val


def sample_pixel_distribution(input_img_fp, sample_number=10e4, std_factor=2):
    """Using input image file path, sample n pixels at random points and then work out a range for scaling across all samples.

    Args:
        input_img_fp (str): Full path to the image (geotiff) to be sampled.
        sample_number (int, optional): How many pixels to sample at random across the image. Defaults to 10e4.
        std_factor (int, optional): How many standard deviations from the mean to use for min/ max range. Defaults to 3.

    Returns:
        list: List of min val, max val tuples one per input image band (colour channel)
    """
    with rio.open(input_img_fp) as f:
        x_min, y_min, x_max, y_max = f.bounds
        coord_list = []
        for i in range(0, int(sample_number)):
            x_point = random.uniform(x_min, x_max)
            y_point = random.uniform(y_min, y_max)
            coord_list.append((x_point, y_point))
        pixel_sample = [val for val in f.sample(coord_list)]
        pixel_arr = np.vstack(pixel_sample)
        return [
            get_min_max_range(pixel_arr[:, j], std_factor) for j in range(0, f.count)
        ]


def rescale_band(img_arr_2d, minmax_tup):
    """Rescale image band 0 to 1, so clipped within specified range.

    Args:
        img_arr_2d (numpy.array): 2-D image array from one band of an image.
        stddev_number (int): Array is clipped so must fall within this many standard deviations of the mean before being normalised.

    Returns:
        numpy.array: Rescaled and normalised array.
    """
    min_val = minmax_tup[0]
    max_val = minmax_tup[1]
    range_val = max_val - min_val
    img_arr_2d = np.where(img_arr_2d < min_val, min_val, img_arr_2d)
    img_arr_2d = np.where(img_arr_2d > max_val, max_val, img_arr_2d)
    img_arr_2d = (img_arr_2d - min_val) / range_val
    img_arr_2d = np.where(img_arr_2d > 1, 1, img_arr_2d)
    return img_arr_2d


def segmentation_preprocessing(img_arr, band_scaling_ranges):
    """Prepare input image array to be used in skimage segmentation algorithm. Assumes B,G,R,X.. image colour channel order.

    Args:
        img_arr (numpy.array): Numpy image array to be scaled.
        band_scaling_ranges (list): Input list of tuples of min max values, one per input array colour channel.

    Returns:
        numpy.array: Input array rescaled using scaling range list
        numpy.array: Just RGB channels rescaled as required by skimage segmentation alg.
    """
    img_arr = img_arr.astype("float64")
    for i in range(0, img_arr.shape[0]):
        img_arr[i] = rescale_band(img_arr[i], band_scaling_ranges[i])
    img_arr_rgb = np.array([img_arr[2], img_arr[1], img_arr[0]])
    img_arr_rgb = np.moveaxis(img_arr_rgb, 0, -1)
    # fig = px.imshow(img_arr_rgb)
    # fig.show()
    np.set_printoptions(precision=3)
    return img_arr, img_arr_rgb


def make_superpixel_img(img_arr, img_arr_rgb):
    """Create a superpixel image using skimage segmentation algorithm Quickshit, taking mean per segment to create superpixels.

    Args:
        img_arr (numpy.array): Full image array of all bands / channels used to create superpixels from segments.
        img_arr_rgb (numpy.array): Red, green, blue array to be used for segmentation.

    Returns:
        Numpy.array: Super pixel image.
    """
    segments = quickshift(
        img_arr_rgb, kernel_size=1, convert2lab=True, max_dist=10, ratio=1
    )
    mean_arr = [
        scipy.ndimage.mean(input=img_arr[i, :, :], labels=segments, index=segments)
        for i in range(0, img_arr.shape[0])
    ]
    mean_arr.append(segments)
    return np.stack(mean_arr)


def write_superpixel_image_windows(input_img_fp, output_img_fp, img_windows):
    """Segment image based on list of window indices, derive superpixels and write to geotiff window-by-window so scalable.

    Args:
        input_img_fp (str): Full path to the image (geotiff) to process.
        output_img_fp (str): Full path to the output image (geotiff) to be created.
        img_windows (list): List of windows to be processed (output of get_windows function).
    """
    with rio.open(input_img_fp) as f:
        prof = f.profile
    # Set dtype, segment compression, count +1 for segment band
    prof.update(dtype="uint16", compress="lzw", count=f.count + 1)
    print("getting image scaling ranges...")
    band_scaling_ranges = sample_pixel_distribution(
        input_img_fp, sample_number=10e4, std_factor=2
    )
    print(band_scaling_ranges)
    out_rio = rio.open(output_img_fp, "w", **prof)
    print(f"{len(img_windows)} windows to process...")
    for i, win_indices in enumerate(img_windows):
        with rio.open(input_img_fp) as f:
            win_arr = f.read(window=Window(*win_indices))
        win_arr, win_arr_rgb = segmentation_preprocessing(win_arr, band_scaling_ranges)
        super_pixel_arr = make_superpixel_img(win_arr, win_arr_rgb)
        # Scale to 16 bit and change dtype
        super_pixel_arr = super_pixel_arr * 10000
        super_pixel_arr = super_pixel_arr.astype("uint16")
        out_rio.write(super_pixel_arr, window=Window(*win_indices))
        if (i + 1) % 10 == 0 or i + 1 == len(img_windows):
            print(f"written window {i+1} of {len(img_windows)}")
    out_rio.close()
    print(f"Completed. See output image {output_img_fp}")


def main(input_img, window_size=1000):
    """Main function to process image, segment tile by tile.

    Args:
        input_img (str): Relative path to image file under the data dir.
        window_size (int, optional): Size of windows processed at a time in pixel dimensions. Defaults to 1000.
    """
    script_dir = Path(__file__).resolve()
    input_img_fp = script_dir.parents[2] / "data" / input_img
    output_name = f"{input_img.replace('.tif','')}_superpixel.tif"
    output_img_fp = script_dir.parents[2] / "data" / output_name
    img_windows = get_windows(input_img_fp, window_size)
    write_superpixel_image_windows(input_img_fp, output_img_fp, img_windows)


def mk_arg_pars():
    """Create a comand line arg parse.
    Returns:
        dict: Argparse argument dictionary.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a superpixel image as mean of pixel values within segments created by quickshift segmentation."
    )
    parser.add_argument(
        "-i",
        "--input-img",
        default="sentinel2-test.tif",
        help="Specify relative path to image file under the data dir. Default sentinel2-test.tif.",
    )

    parser.add_argument(
        "-w",
        "--window-size",
        default=1000,
        help="Specify size of windows to process one-by-one. Default 1000.",
    )

    args_pars = parser.parse_args()
    return vars(args_pars)


if __name__ == "__main__":
    run_dict = mk_arg_pars()
    main(**run_dict)
