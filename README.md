# AI-Powered Flood Evacuation Routing System

A GIS-based hazard-aware evacuation routing prototype that integrates flood risk analysis with real-world road networks to compute safe evacuation routes through an interactive web interface.

---

## Overview

This project combines spatial analysis, flood risk modeling, and graph-based routing to generate safe evacuation paths during flood scenarios.  

It removes flood-prone roads from the network and computes evacuation routes dynamically using a web-based interface.

---

## Features

- Flood risk modeling using rainfall and historical flood data  
- Hazard-aware road classification  
- Safe road graph construction  
- Dijkstra-based evacuation routing  
- Flask REST API for dynamic route computation  
- Interactive Leaflet web map  
- Visualization of unsafe roads and safe routes  

---

## Tech Stack

**Backend**
- Python  
- Flask + Flask-CORS  
- OSMnx  
- NetworkX  

**GIS & Spatial Processing**
- GeoPandas  
- Rasterio  
- Shapely  
- PyProj  
- NumPy  

**Frontend**
- HTML  
- CSS  
- JavaScript  
- Leaflet.js  
- OpenStreetMap  

**Data Formats**
- GeoTIFF  
- GeoJSON  
- GraphML  

---

## Project Structure
FLOOD_EVACUATION_SYSTEM/

├── 1_raw_data/
│
├── 3_outputs/
│   ├── flood_risk.tif
│   ├── unsafe_roads.geojson
│   ├── roads_dehradun_safe.graphml
│   ├── evacuation_map.html
│   └── evacuation_route.geojson
│
├── 4_scripts/
│   └── python/
│       ├── preprocess.py
│       ├── road_osm.py
│       ├── api.py
│       └── map.html
│
└── README.md

