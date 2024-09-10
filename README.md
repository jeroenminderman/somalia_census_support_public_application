<img src="https://github.com/datasciencecampus/awesome-campus/blob/master/ons_dsc_logo.png">

![Python Version](https://img.shields.io/badge/Python-3.9-blue.svg)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![status: experimental](https://github.com/GIScience/badges/raw/master/status/experimental.svg)](https://github.com/GIScience/badges#experimental)

# <Tool Name>

## A tool for population estimation of Internally Displaced People (IDP) camps using Satellite imagery

This repository hosts the code and necessary models to host a streamlit application for population estimation within IDP camps. Given a satellite image and extents of areas containing IDP camps, this tool will predict and output estimated domiciles within the extents given. The outputted `.geojson` files are then usable within your preferred GIS software.

This code was originally written by Nicci Potts, Laurence Jones, Amanda Baizen Edge, and Joshua Onalaja. The project that underpins this code was undertaken in conjunction with the UKs Foreign, Commonwealth, and Development Office (FCDO), Somali government, and United Nations Population Fund (UNFPA) as part of a wider census preparations in Somalia.


## Important information

- The Office for National Statistics (ONS) Data Science Campus (DSC) are currently publishing this tool as an alpha version

- The ONS DSC are `NOT` responsible for any insights gathered from applying the tool to any satellite imagery.

- The output this tool is providing are population estimations and `NOT` exact numbers of tents and/or buildings.

- The ONS DSC will `NOT` guarantee the maintenance of this repository following final release.
  
## How to use this repo

This repository's function is to create footprints from unseen satellite images through a web-based application. Any VHR satellite imagery should be able to be used in this project. Currently the application has been tested with Planet and MAXAR imagery.

The models were developed for this project using Google Cloud Platform (GCP) infrastructure using a NVIDI T4 X 1 notebook with 16 vCPUs, 104 GB RAM, and 1 GPU.

The application has been tested with Python version 3.8/3.9, with all packages provided in `requirements.txt`

A development branch containing the code necessary to build your own models is due to be published soon.

## Project structure tree
Successful running of the application assumes a certain structure in where data and other auxiliary inputs need to be located.
The below tree demonstrates where each file/folder needs to be for successful execution or where files will be located following execution.

### Overview
```
📦somalia_unfpa_census_support
 ┣ 📂data
 ┃ ┣ 📂footprints
 ┃ ┣ 📂imagery
 ┃ ┣ 📂outputs
 ┃ ┗ 📂tiled_images
 ┣ 📂models
 ┃ ┣ 📂conditions
 ┃ ┃ ┗ 📜 <model_name>_<date_and_time>_conditions.txt #subject to change
 ┃ ┗ 📜<model_name>_<date_and_time>.h5 #subject to change
 ┣ 📂src
 ┣ 📂venv-somalia-gcp
 ┣ 📜config.yaml
 ┣ 📜.gitignore
 ┣ 📜requirements.text
 ┗ 📜README.md

```
### Data
```
📦somalia_unfpa_census_support
 ┗ 📂data
   ┣ 📂footprints
   ┃ ┗ 📜<area>_<sub_area>_footprints.geojson
   ┣ 📂imagery
   ┃ ┣ 📂baidoa
   ┃ ┃ ┣ 📂tiles
   ┃ ┃ ┃ ┗ <area>_<sub_area>_camp_extents_polygons_<number>.tif
   ┃ ┃ ┣ 📜<image>.tif
   ┃ ┃ ┗ 📜<area>_<sub_area>_camp_extents.geojson
   ┃ ┗ 📂<area>
   ┣ 📂outputs
   ┗ 📂tiled_images
     ┣ 📂baidoa
     ┃ ┗ 📂<sub_area>.
     ┃   ┣ 📜<area>_<sub_area>_camp_extents_<letter>_<number>_tile_<number>_<number>.tif
     ┃   ┗ 📂polygon_tiles
     ┃     ┗ 📜<area>_<sub_area>_camp_extents_polygons_<number>.tif
     ┗ 📂<area>
```
### Code
```
📦somalia_unfpa_census_support
 ┣ 📂src
 ┃ ┣ 📜.gitkeep
 ┃ ┣ 📜create_footprints.py
 ┃ ┣ 📜create_footprints_functions.py
 ┃ ┣ 📜functions_library.py
 ┃ ┣ 📜homepage.py
 ┃ ┣ 📜loss_functions.py
 ┃ ┣ 📜image_processing_functions.py
 ┃ ┣ 📜multi_class_unet_model_build.py # needed?
 ┃ ┣📂.streamlit
 ┃  ┗ 📜config.toml
 ┃ ┣📂auxiliary-functionality
 ┃  ┣ 📜__init__.py
 ┃  ┗ 📜check_camp_to_image_maps.py
 ┃ ┗📂pages
 ┃  ┣ 📜1_create_tiles.py
 ┃  ┣ 📜2_create_footprints.py
 ┃  ┗ 📜3_save_space.py
 ┣ 📜config.yaml
 ┣ 📜.gitignore
 ┣ 📜requirements.text
 ┗ 📜README.md

```
## Installation

### Virtual environment
Once in the project space (i.e. the base repository level) it is recommended you set-up a virtual environment. In the terminal run:
```
python3 -m venv venv-somalia-app
```
Next, to activate your virtual environment run:
```
venv-somalia-app\Scripts\activate
```
### Install dependencies
While in your active virtual environment, perform a pip install of the `requirements.txt` file, which lists the required dependencies. To do this run:
```
pip install -r requirements.txt
```
With the virtual environment set up, there are two ways to run the application. The first is to type:
```
cd src
streamlit run homepage.py --server.port=8081
```
or the other one is to double click the `run_app.bat` in the base repository folder.

### Note: While you only have to do the install steps once, you will need to activate the virtual environment each time if using the console.

## Camp extent guide

**Please note**
```
- All camp extents will have to be made manually using GIS software. One extent per satellite image is required, however you may wish to draw concise polygons around areas of interest to limit processing requirements.

- These camp extents will need to be exported as `geojsons` to the `\data\imagery\<area>` directory
  before using the app to make tiles and footprints.

- Ensure the `Coordinate Reference System (CRS)` of the created camp extents is set to the same CRS as your satellite image.
```  

### Step by step guide
1) Create and name a directory for the area you want to create footprints for in `\data\imagery`

![image](https://github.com/user-attachments/assets/88ab6713-02b8-427d-8685-28bcaeca7296)

2) Using GIS software create **camp extents** with polygons for your satellite image around the areas of interest.
  
![image](https://github.com/user-attachments/assets/9f09b82f-db98-42a6-ae9c-e42fcb547a84)

3) Export your **camp extents** as `.geojsons` to `\data\imagery\<area>`, **Using a suitable naming scheme like the one shown below.** The model will create sub-area folders based off of the chosen name. Ensure that the relevant satellite images are also placed in this directory.
  
![image](https://github.com/user-attachments/assets/b16178ee-df9b-4479-ae2f-18ad4a0b8448)


### App

**Please note**
```
The app may take some time to load pages after switching between them.
Due to streamlit limitations the app will not complete it's task if you switch tabs on your browser!  
```

The application consists of four pages:

**Homepage** - This is the landing page when opening the application. There is information here detailing the project.

**Create Tiles** - This page looks for folders inside `\data\imagery` that may contain imagery and extents. It will create the required tiled images for the extent areas. The length of time it takes to fulfill this task is dependant on the size of the area required. Please wait for the success message to appear before proceeding to `create footprints`.

**Create Footprints** - This page has you should select the **area** you want to make footprints for, the **model** you want to use. The **sub area** names will correlate directly to a file name given for a set of extents.

Following the success message, go to `\data\footprints` to locate the created footprints. These footprints can then be added to your GIS project and overlaid on your satellite image.

**Clean Up** - The tile creation portion of this application may require multiple gigabytes of storage depending on the size of the satellite images and extents given. This page provides an easy way to remove the intermediary files following footprint creation. However you will need to re-run **Create Tiles** for any area files that are deleted.

### Models <WIP> will update before Tim Leaves.

**Somaliland** - this model was trained on imagery from bossaso and hargesia only

**Full** - this model was trained on all imagery - somaliland and all priority areas

**Tents only** - this model was trained on all imagery somaliland and all priority areas 
but for tents only
