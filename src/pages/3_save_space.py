import streamlit as st
from pathlib import Path

from functions_library import get_folder_paths, rm_tree

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

st.title("Delete tiles")

st.subheader("Here tiles can deleted tiles after footprint creation to save space")

if camp_tiles_dir:
    folders = get_folders_in_directory(camp_tiles_dir)
    if folders:
        selected_folder = st.selectbox("Select region:", folders)
        area_dir = camp_tiles_dir / selected_folder


sub_dirs = get_folders_in_directory(area_dir)
sub_folder = st.selectbox("Select sub area:", sub_dirs)
sub_area_dir = area_dir / sub_folder

st.warning(
    f"""All files and directories in the {sub_area_dir.stem} folder 
        will be removed. If wanting to do this then tick the box below.
        To get these files and directories again please navigate back 
        to the create tiles in the sidebar""", 
        icon="⚠️"
)

save_space = st.checkbox("Remove tiles")

if save_space:
    rm_tree(sub_area_dir)
    st.success(f"All polygon and extent tiles deleted in {sub_area_dir.stem} folder")
