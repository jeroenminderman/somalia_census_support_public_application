from rasterio.mask import mask
from pathlib import Path
import geopandas as gpd
import rasterio as rio
import ipywidgets as widgets
from IPython.display import display
from functions_library import generate_tiles
import streamlit as st
from shapely.geometry import box, Polygon
from shapely.validation import explain_validity
from shapely.errors import TopologicalError
import sys
from functions_library import generate_tiles, get_folder_paths
import time

folder_dict = get_folder_paths()
data_dir = Path(folder_dict["data_dir"])
imagery_dir = Path(folder_dict["imagery_dir"])

st.title("Create Input tiles from Satellite Imagery")
st.subheader(
    "This section of the app will look in the Data folder for relevant satellite imagery"
)

repo_dir = Path.cwd().parent
img_dir = repo_dir.joinpath("images")
st.logo(str(img_dir.joinpath("final_logo.png")))
camp_tiles_dir = Path(folder_dict["camp_tiles_dir"])


def get_folders_in_directory(directory):
    try:
        folders = [item.name for item in directory.iterdir() if item.is_dir()]
        return folders
    except Exception as e:
        st.error(f"an error occured {e}")
        return []


def get_tif_bounding_box(tif_path):
    with rio.open(tif_path) as src:
        bounds = src.bounds
        return box(bounds.left, bounds.bottom, bounds.right, bounds.top), src.crs


def match_tif_to_geojson_polygons(tif_files, geojson_files):
    matches = []

    for geojson_file in geojson_files:
        geojson_gdf = gpd.read_file(geojson_file)

        for idx, polygon in geojson_gdf.iterrows():
            polygon_geom = polygon.geometry
            relevant_tifs = []

            for tif_file in tif_files:
                tif_bbox, tif_crs = get_tif_bounding_box(tif_file)

                if tif_crs != geojson_gdf.crs:
                    tif_bbox = (
                        gpd.GeoSeries([tif_bbox], crs=tif_crs)
                        .to_crs(geojson_gdf.crs)
                        .iloc[0]
                    )

                if polygon_geom.intersects(tif_bbox):
                    relevant_tifs.append(tif_file)

            matches.append((geojson_file, idx, relevant_tifs))
    return matches


def get_intersection_area(tif_path, polygon_geom, geojson_crs):
    with rio.open(tif_path) as src:
        tif_bounds, tif_crs = get_tif_bounding_box(tif_path)

        if tif_crs != geojson_crs:
            tif_bounds = (
                gpd.GeoSeries([tif_bounds], crs=tif_crs).to_crs(geojson_gdf.crs).iloc[0]
            )
        if polygon_geom.is_valid:
            intersection = polygon_geom.intersection(tif_bounds)
            return intersection.area


def match_tif_to_geojson_polygons_2(tif_files, geojson_files):
    matches = []
    for geojson_file in geojson_files:
        geojson_gdf = gpd.read_file(geojson_file)
        geojson_crs = geojson_gdf.crs
        for idx, polygon in geojson_gdf.iterrows():
            if not polygon.geometry.is_empty:
                polygon_geom = polygon.geometry
                relevant_tifs = []
                max_intersection_area = 0.0
                best_tif = 0
                for tif_file in tif_files:
                    intersection_area = get_intersection_area(
                        tif_file, polygon_geom, geojson_crs
                    )
                    if intersection_area:
                        if intersection_area and intersection_area > 0:
                            relevant_tifs.append((tif_file, intersection_area))
                        if intersection_area > max_intersection_area:
                            max_intersection_area = intersection_area
                            best_tif = tif_file
                matches.append(
                    (geojson_file.name, idx, best_tif, max_intersection_area)
                )

    return matches


def create_tile_rasters(polygon, satellite_image, polygon_idx, area_name, tile_dir):
    if not tile_dir.exists():
        tile_dir.mkdir(parents=True, exist_ok=True)

    with rio.open(satellite_image) as src:
        out_image, out_transform = mask(src, [polygon.geometry], crop=True)
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )
        output_file = tile_dir / f"{area_name}_polygons_{polygon_idx}.tif"

        with rio.open(output_file, "w", **out_meta) as dst:
            dst.write(out_image)


def assign_tiles(tif_files, geojson_files, matches, area_folder):
    tile_dir = camp_tiles_dir.joinpath(area_folder.stem)

    progress_text = f"Creating input tiles for {area_folder.stem}"
    progress_bar = st.progress(0, text=progress_text)
    time.sleep(0.01)
    total_iterations = len(geojson_files)
    status_text = st.empty()

    for i, geojson_file in enumerate(geojson_files):
        polygons_gdf = gpd.read_file(geojson_file)

        for idx, polygon in polygons_gdf.iterrows():
            if not polygon.geometry.is_empty:
                polygon_id = idx
                match = next(
                    (
                        item
                        for item in matches
                        if item[0] == geojson_file.name and item[1] == polygon_id
                    ),
                    None,
                )

                if match:
                    satellite_image = ""
                    if match[2] == 0:
                        previous_item = matches[idx - 1]
                        satellite_image = previous_item[2]
                    else:
                        satellite_image = match[2]
                    create_tile_rasters(
                        polygon, satellite_image, idx, geojson_file.stem, tile_dir
                    )
        percent_complete = int((i + 1) / total_iterations * 100)
        progress_bar.progress(percent_complete)
        status_text.text(
            "Progress: {}% - now just generating tiles please wait for success message below".format(
                percent_complete
            )
        )

    for tile_path in tile_dir.glob("*_polygons_*.tif"):
        generate_tiles(tile_path, output_dir=tile_dir)

    st.success(
        f"Tiles successfully created in the folder {area_folder.stem}\ . Please proceed to create footprints"
    )


def check_memory_usage(tif_files):
    """This is here to represent what we could do by loading all satellite images at the same time. The issue is this is highly volatile to the number of satellite images in a folder, and could end up breaking if the memory footprint is too large. For instance, 4 images is roughly 2.5gb"""
    satellite_image_data = {}
    total_memory_usage = 0

    for tif_file in tif_files:
        with rio.open(tif_file) as src:
            image_data = src.read()
            satellite_image_data[tif_file.name] = image_data

            image_memory_usage = image_data.nbytes
            total_memory_usage += image_memory_usage
            total_memory_usage += sys.getsizeof(src.profile) + sys.getsizeof(src.meta)

    st.write(
        f"estimated total memory usage for images: {total_memory_usage / (1024** 2): .2f}MB"
    )


def pipeline(chosen_area):
    tif_files = list(chosen_area.glob("*.tif"))
    geojson_files = list(chosen_area.glob("*.geojson"))
    matches = match_tif_to_geojson_polygons_2(tif_files, geojson_files)

    assign_tiles(tif_files, geojson_files, matches, chosen_area)


def app():
    if imagery_dir:
        folders = get_folders_in_directory(imagery_dir)

        if folders:
            selected_folder = st.selectbox("Select a folder:", folders)

            chosen_area = imagery_dir / selected_folder
            if st.button("Create Tiles"):
                pipeline(chosen_area)


if __name__ == "__main__":
    app()
