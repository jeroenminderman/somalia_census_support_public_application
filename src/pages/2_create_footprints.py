import streamlit as st
import numpy as np
import keras
from keras.models import load_model
from pathlib import Path
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import rasterio as rio
import folium
import shutil
import time


# %%
from functions_library import get_folder_paths
from loss_functions import get_loss_function
from multi_class_unet_model_build import jacard_coef
from data_augmentation_functions import stack_array
from create_footprint_functions import (
    process_tile,
    process_image_files,
    check_shapes,
    filter_empty_arrays,
    extract_transform_from_directory,
    modify_transform,
)

repo_dir = Path.cwd().parent
img_dir = repo_dir.joinpath("images")
st.logo(str(img_dir.joinpath("final_logo.png")))

folder_dict = get_folder_paths()
models_dir = Path(folder_dict["models_dir"])
outputs_dir = Path(folder_dict["outputs_dir"])

camp_tiles_dir = Path(folder_dict["camp_tiles_dir"])
footprints_dir = Path(folder_dict["footprints_dir"])


def get_folders_in_directory(directory):
    try:
        folders = [item.name for item in directory.iterdir() if item.is_dir()]
        return folders
    except Exception as e:
        st.error(f"an error occured {e}")
        return []


def strip_conditions(text):
    pairs = text.split(",")
    n_classes_value = None

    for pair in pairs:
        key, value = pair.split("=")
        key = key.strip()
        value = value.strip()

        if key == "n_classes":
            n_classes_value = value
            return n_classes_value


def sort_camp_tiles(area_dir):
    img_files = list(area_dir.glob("*.tif"))
    for file in img_files:
        second_word = file.stem.split("_")[1]

        destination_dir = area_dir / second_word
        destination_dir.mkdir(parents=True, exist_ok=True)
        polygon_dir = destination_dir / "polygon_tiles"
        polygon_dir.mkdir(exist_ok=True)

        if "polygons" in file.stem:
            shutil.move(str(file), str(polygon_dir / file.name))
        else:
            shutil.move(str(file), str(destination_dir / file.name))


def build_arrays(sub_area_dir):
    ### ADD TRY/EXCEPT HERE
    img_files = list(sub_area_dir.glob("*.tif"))
    # Read the first GeoTIFF file
    first_tiff_file = img_files[0]

    # Open the GeoTIFF file
    with rio.open(first_tiff_file) as src:
        # Get CRS
        crs = src.crs

    img_size = 384
    error_files = process_image_files(img_files, img_size, sub_area_dir)
    check_shapes(sub_area_dir)

    unseen_images, unseen_filenames = stack_array(sub_area_dir, expanded_outputs=True)

    # for file in Path(area_dir).glob("*npy"):
    # file.unlink()

    # ### Add padding
    padding = 8
    padded_unseen_images = np.pad(
        unseen_images,
        ((0, 0), (padding, padding), (padding, padding), (0, 0)),
        mode="constant",
    )
    st.write(f"{len(padded_unseen_images)} arrays created")

    # ### Check for any blank arrays
    filtered_images, filtered_filenames = filter_empty_arrays(
        padded_unseen_images, unseen_filenames
    )
    # create npy arrays & highlight any files with issues
    error_files = process_image_files(img_files, img_size, sub_area_dir)

    return filtered_images, filtered_filenames, crs


def create_polygons(
    model, filtered_images, filtered_filenames, sub_area_dir, n_classes, crs, sub_folder
):
    # get transformation matrix from original .tiff images
    transforms = extract_transform_from_directory(sub_area_dir)
    padding = 8

    modified_transforms = dict(
        (k, modify_transform(v, padding)) for k, v in transforms.items()
    )
    unique_classes = list(range(n_classes))
    all_results = []

    progress_text = "Building Footprints. Please wait."
    progress_bar = st.progress(0, text=progress_text)
    total_iterations = len(filtered_images)
    status_text = st.empty()

    for idx, (tile, filename) in enumerate(zip(filtered_images, filtered_filenames)):
        result_gdf = process_tile(
            model, tile, unique_classes, filename, idx, modified_transforms, crs
        )
        if result_gdf is not None:
            all_results.append(result_gdf)
        percent_complete = int((idx + 1) / total_iterations * 100)
        progress_bar.progress(percent_complete)
        status_text.text("Progress: {}%".format(percent_complete))

    all_polygons_gdf = pd.concat(all_results, ignore_index=True)
    all_polygons_gdf["size"] = all_polygons_gdf["geometry"].area
    all_polygons_gdf = all_polygons_gdf[all_polygons_gdf["size"] >= 1]
    all_polygons_gdf.reset_index(inplace=True, drop=True)
    
    footprints_dir.mkdir(exist_ok=True)
    output_footprints = footprints_dir / f"{sub_folder}_footprints.geojson"
    all_polygons_gdf.to_file(output_footprints, driver="GeoJSON")

    with st.spinner("Waiting..."):
        time.sleep(1)

    st.success(
        f"Footprints created for structures in {sub_folder}. They can be found in the {footprints_dir}",
        icon="✅",
    )


def pipeline(area_dir, runid):
    conditions_path = models_dir / f"conditions/{runid}_conditions.txt"
    with open(conditions_path, "r") as file:
        text = file.read()

    n_classes = int(
        strip_conditions(text)
    )  ## FIX OUR OUTPUT IN TRAIN_NOTEBOOK. MISSING A COMMA AFTER EPOCHS = X
    # Need to update this to allow other options / generate it from the model outputs better
    loss = get_loss_function("combined")

    model_filename = f"{runid}.h5"
    model_phase = models_dir / model_filename

    model = keras.models.load_model(
        model_phase,
        custom_objects={
            "focal_loss": loss,
            "dice_loss": loss,
            "jacard_coef": jacard_coef,
        },
    )

    sort_camp_tiles(area_dir)
    sub_dirs = get_folders_in_directory(area_dir)
    sub_folder = st.selectbox("Select a sub area to build footprints:", sub_dirs)
    sub_area_dir = area_dir / sub_folder

    if st.button("Create Footprints"):
        st.write(f"Building Arrays from tiles in {sub_area_dir}")
        final_images, final_filenames, crs = build_arrays(sub_area_dir)
        st.write("Creating Polygons...")
        create_polygons(
            model,
            final_images,
            final_filenames,
            sub_area_dir,
            n_classes,
            crs,
            sub_folder,
        )

    return


def app():
    if camp_tiles_dir:
        folders = get_folders_in_directory(camp_tiles_dir)
        if folders:
            selected_folder = st.selectbox("Select a folder:", folders)
            area_dir = camp_tiles_dir / selected_folder
            
            model_list = [item.stem for item in models_dir.iterdir() if item.is_file() and item.name != '.gitkeep']
            selected_model = st.selectbox(
                "Select the model you wish to create footprints from", model_list
            )

            # if st.button("Load Model"):
            pipeline(area_dir, selected_model)


if __name__ == "__main__":
    app()
