from create_super_pixel_image import get_windows
from utils.predictions_to_shape import classified_img_to_shp, dissolve_contiguous
import argparse
import geopandas as gpd
from shapely.geometry import box
import pandas as pd
import rasterio as rio
from rasterio.windows import Window
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from pathlib import Path

def read_reproject_training(sample_points_fp, input_img_fp):
    """_summary_

    Args:
        sample_points_fp (str): Full or relative path to the shapefile to be reprojected.
        input_img_fp (str): Full or relative path to raster having the crs to reproject shapefile to.

    Returns:
        geopandas.GeoDataFrame: Reprojected geodataframe matching raster crs.
    """
    with rio.open(input_img_fp) as f:
        raster_crs = f.profile["crs"].to_string()
    sample_points = gpd.read_file(sample_points_fp)
    return sample_points.to_crs(raster_crs, inplace=True)


def extract_pixel_training_points(input_img_fp, sample_points_fp):
    """Extract pixel values training point shape locations and build numpy array for model training.

    Args:
        input_img_fp (str): Path to image file to extract pixel values from.
        sample_points_fp (str): Path to point shapefile to extract pixel values from.

    Returns:
        numpy.array: Pixel values in rows.
        numpy.array: Label classes in rows.
    """
    # Read and reproject training data to same crs as raster
    sample_points = read_reproject_training(sample_points_fp, input_img_fp)
    # filter training points by image extent
    with rio.open(input_img_fp) as f:
        img_bounds = f.bounds
        crs = f.crs
    img_box = gpd.GeoDataFrame(geometry=[box(*img_bounds)], crs=crs)
    sample_points = sample_points[sample_points.within(img_box.unary_union)]
    # Separate sample points into classes, locations of tree, non-tree segments
    tree_samples = sample_points[sample_points["ml_class"] == 1]
    non_tree_samples = sample_points[sample_points["ml_class"] == 0]
    tree_coord_list = [
        f for f in zip(list(tree_samples.geometry.x), list(tree_samples.geometry.y))
    ]
    non_tree_coord_list = [
        f
        for f in zip(
            list(non_tree_samples.geometry.x), list(non_tree_samples.geometry.y)
        )
    ]
    # Take pixel values at sample point locations
    with rio.open(input_img_fp) as f:
        b = f.count - 1
        tree_pixel_samples = [val[0:b] for val in f.sample(tree_coord_list)]
        non_tree_pixel_samples = [val[0:b] for val in f.sample(non_tree_coord_list)]
    # Build the np arrays for ML
    X = np.vstack(tree_pixel_samples + non_tree_pixel_samples)
    # X = calc_ndvi(X, 2, 3)
    y = np.append(
        np.ones(len(tree_pixel_samples)), np.zeros(len(non_tree_pixel_samples))
    )
    return X, y


def test_rf(X, y, estimators=100):
    """Split training data into train-test, train rf and print prediction metrics against test set.

    Args:
        X (numpy.Array): Numpy.array of training data for random forest.
        y (numpy.Array): Numpy.array of binary labels for random forest.
        estimators (int): How many decision trees in random forest.
    """
    trainX, testX, trainy, testy = train_test_split(X, y, stratify=y)
    print(
        f"-->trainX shape {trainX.shape}\n-->testX shape {testX.shape}\n-->trainy shape {trainy.shape}\n-->testy shape {testy.shape}"
    )
    rf = RandomForestClassifier(n_estimators=estimators)
    print("fitting Random Forest classifier on training set...")
    rf.fit(trainX, trainy)
    predy = rf.predict(testX)
    cm = confusion_matrix(testy, predy)
    acc = accuracy_score(testy, predy)
    f1 = f1_score(testy, predy)
    print(
        f"RF test set results:\nConfusion:\n{cm}\nOverall: {round(acc*100, 3)}%\nF1: {round(f1,3)}"
    )


def calc_ndvi(features_arr, red_idx, nir_idx):
    """Calculate an NDVI column using the red and nir columns of a 2-d feature array.

    Args:
        features_arr (numpy.array): Input must be shaped for ML - rows of pixels, columns of pixel values.
        red_idx (integer): Index of the red column in axis=1
        nir_idx (integer): Index of the nir column in axis=2

    Returns:
        numpy.arr: Input arr with extra column for NDVI added.
    """
    ndvi = (features_arr[:, nir_idx] - features_arr[:, red_idx]) / (
        features_arr[:, nir_idx] + features_arr[:, red_idx]
    )
    ndvi = ndvi.reshape(features_arr.shape[0], 1)
    return np.append(features_arr, ndvi, axis=1)


def rf_predict_img_win(win_arr, trained_classifier, prob=True):
    """Predict image window using input trained classifier.

    Args:
        win_arr (numpy.arr): In rasterio order (channels, y, x)
        trained_classifier (sklearn.model): Trained sklearn model to use for predictions.
        prob (bool, optional): Generate probability of prediction or binary prediction. Defaults to True.

    Returns:
        numpy.arr: Array of predictions.
    """

    # Get dims
    b, y, x = win_arr.shape
    segment_idx = b - 1

    # Reshape for classifier
    win_arr = np.transpose(win_arr.reshape(b, -1))

    img_bnds = [i for i in range(0, b) if i != segment_idx]
    win_arr = win_arr[:, img_bnds]
    # No data rows
    no_data = np.any(win_arr, axis=1).astype("uint8")
    # Calc ndvi
    # win_arr = calc_ndvi(win_arr, 2, 3)

    # Prob predictions
    if prob:
        pred_arr = trained_classifier.predict_proba(win_arr)
        # subset just the positive (forest) class probaility for all pixels
        pred_arr = pred_arr[:, 1:]

    # Or class predictions
    else:
        pred_arr = trained_classifier.predict(win_arr)
    # Reshape back to image
    pred_arr = pred_arr.reshape(y, x)
    no_data = no_data.reshape(y, x)
    # Apply no data mask so not positive prediction
    pred_arr = pred_arr * no_data
    return pred_arr


def predict_write_segmented_img_windows(
    windows_list, input_img_fp, output_shp_fp, trained_classifier, class_threshold=0.7
):
    """Make predictions for each window in specified list, convert to geometries, write all to shapefile.

    Args:
        windows_list (list): List of image window indices to process.
        input_img_fp (str): Full path to image file to process.
        output_shp_fp (str): Full path to output results shapefile
        trained_classifier (sklearn.classifier): Trained sklearn model used to make predictions.
        class_threshold (float): Threshold above which to make positive prediction. Default 0.7.
    """
    with rio.open(input_img_fp) as f:
        prof = f.profile
    prof.update(count=1)
    gdf_list = []
    for i, win_indices in enumerate(windows_list):
        with rio.open(input_img_fp) as f:
            current_window = Window(*win_indices)
            win_arr = f.read(window=current_window)
            win_transform = f.window_transform(current_window)
            prof.update(
                transform=win_transform, width=win_arr.shape[2], height=win_arr.shape[1]
            )
        pred_arr = rf_predict_img_win(win_arr, trained_classifier, prob=True)
        pred_arr = (pred_arr > class_threshold).astype("uint8")
        pred_arr = pred_arr.astype("uint8")
        pred_shape = classified_img_to_shp(pred_arr, prof)
        if pred_shape is not None:
            gdf_list.append(pred_shape)
        if (i + 1) % 10 == 0 or i + 1 == len(windows_list):
            print(f"predicted window {i+1} of {len(windows_list)}")
    if len(gdf_list) > 0:
        out_gdf = gpd.GeoDataFrame(
            pd.concat(gdf_list, ignore_index=True), crs=gdf_list[0].crs
        )
        out_gdf = dissolve_contiguous(out_gdf)
        out_gdf.to_file(output_shp_fp, index=False)
        print(f"Completed. See prediction shapefile {output_shp_fp}")
    else:
        print("no prediction results to be written.")


def main(input_img):
    """Main process generating predictions for the input image.

    Args:
        input_img (str): Specify relative path to superpixel image file under the data dir.
    """
    # set paths
    script_dir = Path(__file__).resolve()
    input_img_fp = script_dir.parents[2] / "data" / input_img
    sample_points_fp = script_dir.parents[2] / "input" / "segment_sample_points.gpkg"
    output_name = f"{input_img.replace('.tif', '')}_pred.shp"
    if "/" in output_name:
        output_name = output_name.split("/")[-1]
    output_shp_fp = script_dir.parents[2] / "data" / "prediction_shapes" / output_name

    # build X, y
    X, y = extract_pixel_training_points(input_img_fp, sample_points_fp)
    # Test RF classifier
    test_rf(X, y)
    # Fit classifier on full sample point set
    trained_classifier = RandomForestClassifier(n_estimators=100)
    trained_classifier.fit(X, y)

    # Create window indices
    windows_list = get_windows(input_img_fp)
    predict_write_segmented_img_windows(
        windows_list, input_img_fp, output_shp_fp, trained_classifier
    )

    # Convert whole image prediction to shape


def mk_arg_pars():
    """Create a comand line arg parse.
    Returns:
        dict: Argparse argument dictionary.
    """

    parser = argparse.ArgumentParser(
        description="Create Random Forest predictions from superpixel image and labelled training points"
    )
    parser.add_argument(
        "-i",
        "--input-img",
        default="sentinel2-superpixel-test.tif",
        help="Specify relative path to superpixel image file under the data dir. Default sentinel2-superpixel-test.tif.",
    )

    args_pars = parser.parse_args()
    return vars(args_pars)


if __name__ == "__main__":
    run_dict = mk_arg_pars()
    main(**run_dict)
