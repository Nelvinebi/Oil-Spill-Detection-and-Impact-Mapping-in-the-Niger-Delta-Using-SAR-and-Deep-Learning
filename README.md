🛢️ Oil Spill Detection and Impact Mapping in the Niger Delta Using SAR and Deep Learning

This project implements a deep learning–based framework for detecting oil spills and mapping their environmental impact using synthetic Synthetic Aperture Radar (SAR) data. A U-Net convolutional neural network is employed for pixel-level oil spill segmentation, with outputs exported to GIS-ready formats (GeoTIFF and Shapefile) for spatial analysis.

📌 Features

Realistic synthetic SAR data generation mimicking Sentinel-1 backscatter behavior

Oil spill simulation based on low-backscatter, smooth surface properties

U-Net deep learning model for oil spill segmentation

Model evaluation using standard classification metrics

GIS export of predicted oil spill impact areas (GeoTIFF & Shapefile)

Fully compatible with QGIS and ArcGIS

🧠 Methodology

Synthetic SAR Generation
Simulates water and oil spill backscatter characteristics observed in SAR imagery.

Deep Learning Segmentation
A U-Net architecture performs pixel-wise classification of oil spill regions.

Impact Mapping
Predicted oil spill masks are converted into georeferenced raster and vector GIS layers.

📁 Project Structure
├── oil_spill_detection_sar_unet.py
├── synthetic_sar_oil_spill_dataset.csv
├── oil_spill_impact_map.tif
├── oil_spill_impact_shapefile.shp
├── README.md

🛠️ Requirements

Python 3.8+

NumPy

SciPy

Matplotlib

scikit-learn

TensorFlow / Keras

rasterio

fiona

Install dependencies:

pip install numpy scipy matplotlib scikit-learn tensorflow rasterio fiona

▶️ How to Run
python oil_spill_detection_sar_unet.py


The script will:

Generate synthetic SAR images

Train the U-Net model

Evaluate segmentation performance

Visualize oil spill predictions

🌍 GIS Outputs

GeoTIFF: Binary oil spill impact raster

Shapefile: Vectorized oil spill polygons for spatial analysis

These outputs can be directly loaded into QGIS or ArcGIS for further environmental assessment.

📊 Applications

Environmental monitoring in the Niger Delta

Oil spill impact assessment

Remote sensing and GIS research

Academic projects and theses

Deep learning demonstrations using SAR data

Author: 
Agbozu Ebingiye Nelvin

Email: nelvinebingiye@gmail.com

GitHub: *https://github.com/Nelvinebi

LinkedIn: *https://www.linkedin.com/in/agbozu-ebi/


⚠️ Disclaimer

This project uses synthetic data for research and educational purposes. Results should not be used for operational decision-making without validation using real satellite data.

📜 License

This project is released under the MIT License.
