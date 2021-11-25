import ee
import time
import geemap
import geopandas as gpd
from pathlib import Path
import webbrowser

from uganda_forestry.src.python.train_predict_rf import export_prediction_to_tif

ee.Initialize()


def get_s2c(aoi, start_date, end_date, img_cloud_filter):
    """Return GEE Sentinel 2 image collection with s2cloudless cloud prob band.

    Args:
        aoi (ee.Geometry): Google Earth Engine geometry bounding box for area of interest
        start_date (date (yyyy-MM-dd)): Min date for image collection images
        end_date (date (yyyy-MM-dd)): Max data for image collection images
        img_cloud_filter (int): Max amount of cloud in image metadata

    Returns:
        ee.ImageCollection : Filtered image collection of joined s2 and s2cloudless images.
    """
    # Import and filter S2 SR.
    s2c = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", img_cloud_filter))
    )

    # Import and filter s2cloudless.
    s2_cloud = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )

    innerJoin = ee.Join.inner()
    filterID = ee.Filter.equals(leftField="system:index", rightField="system:index")

    innerJoined = innerJoin.apply(s2c, s2_cloud, filterID)

    return ee.ImageCollection(
        innerJoined.map(lambda f: ee.Image.cat(f.get("primary"), f.get("secondary")))
    )


def get_s2_img(
    extent_fc,
    start_date,
    end_date,
    cloud_img_whole=15,
    cloud_pixel_prob=25,
    s2_bands=["B2", "B3", "B4", "B8", "B11"],
):
    """Create s2 image with selected bands for aoi.

    Args:
        extent_fc (ee.Geometry()): bounding box for image
        date_from (date (yyyy-MM-dd)): date min for composite image
        date_to (date (yyyy-MM-dd)): date max for composite image
        cloud_img_whole (int, optional): s2 image metadata whole cloudy pixel max percent. Defaults to 15.
        cloud_pixel_prob (int, optional): s2cloudless pixel probability. Defaults to 25.

    Returns:
        ee.Image
    """
    s2c = get_s2c(extent_fc, start_date, end_date, cloud_img_whole)

    def mask_cloud(img):
        mask = img.select("probability").focal_min(2).focal_max(10).lt(cloud_pixel_prob)
        return img.updateMask(mask)

    s2c = s2c.map(mask_cloud)
    s2_img = s2c.median().select(s2_bands)

    return s2_img


def get_s1_img(extent_fc, start_date, end_date, bands=["VH", "VV"]):
    """Return a Sentinel 1 SAR temporal composite image.

    Args:
        extent_fc (ee.Geometry): bounding box for image
        date_from (date (yyyy-MM-dd)): date min for composite image
        date_to (date (yyyy-MM-dd)): date max for composite image
        bands (list, optional): Selected Sentinel 1 bands of VV and VH. Defaults to ["VH"].

    Returns:
        ee.Image: Composite ee.Image for Sentinel 1 image
    """
    s1c = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(extent_fc)
        .filterDate(start_date, end_date)
    )
    return s1c.median().select(bands)


def s1_add_ratio(input_image):
    """Sentinel 1 ratio band calculated by subtracting VV and VH polarisations.

    Args:
        input_image (ee.Image): Sentinel 1 GEE image with VV and VH bands.

    Returns:
        ee.Image: Sentinel 1 ee.Image with VV/VH ratio band added.
    """
    return input_image.addBands(
        input_image.select("VV").subtract(input_image.select("VH")).rename("ratio")
    )


def export_img_gdrive(
    gee_img, folder_name, img_file_name, aoi, crs="EPSG:20135", poll_completion=False
):
    task = ee.batch.Export.image.toDrive(
        image=gee_img,
        folder=folder_name,
        description=img_file_name,
        region=aoi,
        scale=10,
        crs=crs,
        maxPixels=10000000000,
    )
    task.start()
    if poll_completion:
        while task.active():
            print("Polling for task (id: {}).".format(task.id))
            time.sleep(30)
        print("Done with training export.")


def get_shape_geometry_crs(input_shape_fp):
    shape_gdf = gpd.read_file(input_shape_fp)
    original_crs = shape_gdf.crs.to_string()
    shape_gdf.to_crs("EPSG:4326", inplace=True)
    aoi_geom = ee.Geometry.Rectangle(*shape_gdf.total_bounds)
    return original_crs, aoi_geom


def show_gee_map(aoi_geom, image_to_show, data_dir, sentinel_satellite="sentinel2"):
    # Make the template map
    preview_map = geemap.Map()
    preview_map.centerObject(aoi_geom, zoom=10)

    # Add the image layers with appropriate visualisation parameters
    if sentinel_satellite == "sentinel2":
        vis_params = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 2000}
        preview_map.addLayer(image_to_show, vis_params, "sentinel2")
    elif sentinel_satellite == "sentinel1":
        vis_params = {
            "bands": ["VV", "VH", "ratio"],
            "min": [-15, -20, 0],
            "max": [0, -5, 15],
        }
        preview_map.addLayer(image_to_show, vis_params, "sentinel1")

    # Save the map html temporarily (only way to display geemap outside of Jupyter Notebook)
    out_html = data_dir / "preview_map.html"
    preview_map.save(out_html)
    webbrowser.open(f"file://{out_html}")
    export_response = input("continue and export image y/n?")
    if export_response.lower() in ("y", "yes"):
        return None
    else:
        print("Selected not to export")
        quit()


def main(
    input_shape="sentinel2_aoi.shp",
    sentinel_satellite="sentinel2",
    start_date="2020-04-01",
    end_date="2020-05-31",
    plot_before_export=True,
):
    # Set directories
    script_path = Path(__file__).resolve()
    data_dir = script_path.parents[2] / "data"
    input_shape_fp = data_dir / input_shape

    # Get the area of interest as GEE geometry
    output_crs, aoi_geom = get_shape_geometry_crs(input_shape)

    # Get the sentinel image based on requested type
    if sentinel_satellite == "sentinel2":
        output_img = get_s2_img(aoi_geom, start_date, end_date)
    elif sentinel_satellite == "sentinel1":
        output_img = get_s1_img(aoi_geom, start_date, end_date)
    else:
        raise ValueError(
            "Must specifiy sentinel2 or sentinel1 for sentinel_satellite parameter"
        )

    # Show map of image before export
    if plot_before_export:
        show_gee_map(aoi_geom, output_img, data_dir, sentinel_satellite)
    # TO DO add exporting
    export_img_gdrive(output_img, "")
