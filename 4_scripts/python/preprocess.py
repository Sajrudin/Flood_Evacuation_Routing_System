import os
os.environ["SHAPE_RESTORE_SHX"] = "YES"
import numpy as np
import geopandas as gpd

imd_path = r"S:\Work\Flood_Evacuation_System\1_raw_data\IMD_Station.shp"
imd = gpd.read_file(imd_path)

# Reproject to UTM 44N
imd_utm = imd.to_crs(epsg=32644)

print("Original CRS:")
print(imd.crs)

print("\nReprojected CRS:")
print(imd_utm.crs)

print(imd_utm[["Station", "RF_1Day"]].head())

#  Clean rainfall data
imd_utm = imd_utm[imd_utm["RF_1Day"].notnull()]
imd_utm = imd_utm[imd_utm["RF_1Day"] > 0]

print("Number of valid IMD stations:", len(imd_utm))

# Create study area extent from IMD points

minx, miny, maxx, maxy = imd_utm.total_bounds

# add 10 km buffer on all sides
buffer = 10_000  # meters

extent = {
    "minx": minx - buffer,
    "miny": miny - buffer,
    "maxx": maxx + buffer,
    "maxy": maxy + buffer
}

print("Study area extent (EPSG:32644):")
print(extent)

#  Create raster grid

import numpy as np

resolution = 1000  # grid size in meters (1 km)

x_coords = np.arange(extent["minx"], extent["maxx"], resolution)
y_coords = np.arange(extent["miny"], extent["maxy"], resolution)

print("Grid info:")
print("X cells:", len(x_coords))
print("Y cells:", len(y_coords))
print("Total grid cells:", len(x_coords) * len(y_coords))

#  Create grid coordinate pairs

xx, yy = np.meshgrid(x_coords, y_coords)

grid_points = np.column_stack([xx.ravel(), yy.ravel()])

print("Grid points shape:", grid_points.shape)
print("First 5 grid points:")
print(grid_points[:5])

#  Prepare IMD station arrays for IDW

# Extract station coordinates
station_coords = np.column_stack([
    imd_utm.geometry.x.values,
    imd_utm.geometry.y.values
])

# Extract rainfall values
rainfall_values = imd_utm["RF_1Day"].values

print("Station coords shape:", station_coords.shape)
print("Rainfall values shape:", rainfall_values.shape)

print("First 5 stations (x, y, rainfall):")
for i in range(5):
    print(station_coords[i], rainfall_values[i])

#  IDW interpolation (chunked)

from scipy.spatial import cKDTree

def idw_chunked(station_xy, values, grid_xy, power=2, k=8, chunk_size=200_000):
    tree = cKDTree(station_xy)
    result = np.empty(len(grid_xy), dtype="float32")

    for i in range(0, len(grid_xy), chunk_size):
        chunk = grid_xy[i:i + chunk_size]
        dist, idx = tree.query(chunk, k=k)

        weights = 1 / (dist ** power)
        weights[dist == 0] = 1e12

        weighted_vals = np.sum(weights * values[idx], axis=1)
        result[i:i + chunk_size] = weighted_vals / np.sum(weights, axis=1)

        print(f"Processed {i + len(chunk)} / {len(grid_xy)} grid points", end='\r')

    return result


print("Starting IDW interpolation...")
rainfall_idw = idw_chunked(
    station_coords,
    rainfall_values,
    grid_points,
    power=2,
    k=8
)

print("IDW interpolation finished.")
print("Rainfall stats:")
print("Min:", rainfall_idw.min())
print("Max:", rainfall_idw.max())

#  Save rainfall IDW raster
import rasterio
from rasterio.transform import from_origin

# Reshape to raster (rows = Y, cols = X)
rainfall_raster = rainfall_idw.reshape(len(y_coords), len(x_coords))

# Define raster transform
transform = from_origin(
    extent["minx"],
    extent["maxy"],
    resolution,
    resolution
)

output_raster = r"S:\Work\Flood_Evacuation_System\3_outputs\rainfall_idw.tif"

with rasterio.open(
    output_raster,
    "w",
    driver="GTiff",
    height=rainfall_raster.shape[0],
    width=rainfall_raster.shape[1],
    count=1,
    dtype=rainfall_raster.dtype,
    crs="EPSG:32644",
    transform=transform,
) as dst:
    dst.write(rainfall_raster, 1)

print("Rainfall IDW raster saved at:")
print(output_raster)

#  Verify saved raster

with rasterio.open(output_raster) as src:
    print("Raster CRS:", src.crs)
    print("Raster width, height:", src.width, src.height)
    print("Raster bounds:", src.bounds)

    data = src.read(1)
    print("Raster min:", data.min())
    print("Raster max:", data.max())


#  Verify flood rasters (inspection only)

flood_files = {
    "2008": r"S:\Work\Flood_Evacuation_System\1_raw_data\flood_rasters\flood_2008.tif",
    "2009": r"S:\Work\Flood_Evacuation_System\1_raw_data\flood_rasters\flood_2009.tif",
    "2010": r"S:\Work\Flood_Evacuation_System\1_raw_data\flood_rasters\flood_2010.tif",
}

for year, path in flood_files.items():
    with rasterio.open(path) as src:
        data = src.read(1)

        print(f"\nFlood raster {year}:")
        print(" CRS:", src.crs)
        print(" Width, Height:", src.width, src.height)
        print(" Min value:", data.min())
        print(" Max value:", data.max())


#Reproject flood_2008 to match rainfall raster

from rasterio.warp import reproject, Resampling

flood_2008_path = r"S:\Work\Flood_Evacuation_System\1_raw_data\flood_rasters\flood_2008.tif"
flood_2008_out = r"S:\Work\Flood_Evacuation_System\3_outputs\flood_2008_utm.tif"

with rasterio.open(output_raster) as ref:  # rainfall raster as reference
    ref_transform = ref.transform
    ref_crs = ref.crs
    ref_width = ref.width
    ref_height = ref.height

with rasterio.open(flood_2008_path) as src:
    flood_2008_data = src.read(1)

    dest = np.zeros((ref_height, ref_width), dtype="uint8")

    reproject(
        source=flood_2008_data,
        destination=dest,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.nearest
    )

with rasterio.open(
    flood_2008_out,
    "w",
    driver="GTiff",
    height=ref_height,
    width=ref_width,
    count=1,
    dtype="uint8",
    crs=ref_crs,
    transform=ref_transform,
) as dst:
    dst.write(dest, 1)

print("Flood 2008 reprojected and saved at:")
print(flood_2008_out)

#  Verify reprojected flood_2008 raster

flood_2008_utm = r"S:\Work\Flood_Evacuation_System\3_outputs\flood_2008_utm.tif"

with rasterio.open(flood_2008_utm) as src:
    data = src.read(1)

    print("Flood 2008 UTM CRS:", src.crs)
    print("Flood 2008 UTM width, height:", src.width, src.height)
    print("Flood 2008 UTM min:", data.min())
    print("Flood 2008 UTM max:", data.max())

#  Reproject flood_2009 and flood_2010 to match rainfall raster

from rasterio.warp import reproject, Resampling

flood_inputs = {
    "2009": r"S:\Work\Flood_Evacuation_System\1_raw_data\flood_rasters\flood_2009.tif",
    "2010": r"S:\Work\Flood_Evacuation_System\1_raw_data\flood_rasters\flood_2010.tif",
}

with rasterio.open(output_raster) as ref:
    ref_transform = ref.transform
    ref_crs = ref.crs
    ref_width = ref.width
    ref_height = ref.height

for year, in_path in flood_inputs.items():
    out_path = rf"S:\Work\Flood_Evacuation_System\3_outputs\flood_{year}_utm.tif"

    with rasterio.open(in_path) as src:
        src_data = src.read(1)
        dest = np.zeros((ref_height, ref_width), dtype="uint8")

        reproject(
            source=src_data,
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.nearest
        )

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=ref_height,
        width=ref_width,
        count=1,
        dtype="uint8",
        crs=ref_crs,
        transform=ref_transform,
    ) as dst:
        dst.write(dest, 1)

    print(f"Flood {year} reprojected and saved at:")
    print(out_path)

#  Create flood frequency raster (2008–2010)

flood_2008_utm = r"S:\Work\Flood_Evacuation_System\3_outputs\flood_2008_utm.tif"
flood_2009_utm = r"S:\Work\Flood_Evacuation_System\3_outputs\flood_2009_utm.tif"
flood_2010_utm = r"S:\Work\Flood_Evacuation_System\3_outputs\flood_2010_utm.tif"

with rasterio.open(flood_2008_utm) as f08, \
     rasterio.open(flood_2009_utm) as f09, \
     rasterio.open(flood_2010_utm) as f10:

    data_08 = f08.read(1)
    data_09 = f09.read(1)
    data_10 = f10.read(1)

    flood_frequency = data_08 + data_09 + data_10

    ref_meta = f08.meta.copy()
    ref_meta.update({
        "dtype": "uint8",
        "count": 1
    })

output_freq = r"S:\Work\Flood_Evacuation_System\3_outputs\flood_frequency_2008_2010.tif"

with rasterio.open(output_freq, "w", **ref_meta) as dst:
    dst.write(flood_frequency.astype("uint8"), 1)

print("Flood frequency raster saved at:")
print(output_freq)

#  Verify flood frequency raster

freq_path = r"S:\Work\Flood_Evacuation_System\3_outputs\flood_frequency_2008_2010.tif"

with rasterio.open(freq_path) as src:
    data = src.read(1)

    print("Flood frequency CRS:", src.crs)
    print("Flood frequency width, height:", src.width, src.height)
    print("Flood frequency min:", data.min())
    print("Flood frequency max:", data.max())
    print("Flood frequency unique values:", sorted(set(data.flatten())))

#  Normalize flood frequency raster (0–1)

freq_path = r"S:\Work\Flood_Evacuation_System\3_outputs\flood_frequency_2008_2010.tif"
freq_norm_out = r"S:\Work\Flood_Evacuation_System\3_outputs\flood_frequency_norm.tif"

with rasterio.open(freq_path) as src:
    freq = src.read(1).astype("float32")
    meta = src.meta.copy()

# Normalize (max possible value = 3)
freq_norm = freq / 3.0

meta.update({
    "dtype": "float32",
    "count": 1
})

with rasterio.open(freq_norm_out, "w", **meta) as dst:
    dst.write(freq_norm, 1)

print("Normalized flood frequency raster saved at:")
print(freq_norm_out)

print("Normalized min:", freq_norm.min())
print("Normalized max:", freq_norm.max())

# Normalize rainfall raster (0–1)

rainfall_path = r"S:\Work\Flood_Evacuation_System\3_outputs\rainfall_idw.tif"
rainfall_norm_out = r"S:\Work\Flood_Evacuation_System\3_outputs\rainfall_norm.tif"

with rasterio.open(rainfall_path) as src:
    rain = src.read(1).astype("float32")
    meta = src.meta.copy()

# Min-Max normalization
rain_min = rain.min()
rain_max = rain.max()
rain_norm = (rain - rain_min) / (rain_max - rain_min)

meta.update({
    "dtype": "float32",
    "count": 1
})

with rasterio.open(rainfall_norm_out, "w", **meta) as dst:
    dst.write(rain_norm, 1)

print("Normalized rainfall raster saved at:")
print(rainfall_norm_out)
print("Normalized rainfall min:", rain_norm.min())
print("Normalized rainfall max:", rain_norm.max())

#  Create Flood Risk Map (weighted overlay)

flood_norm_path = r"S:\Work\Flood_Evacuation_System\3_outputs\flood_frequency_norm.tif"
rain_norm_path = r"S:\Work\Flood_Evacuation_System\3_outputs\rainfall_norm.tif"
risk_out = r"S:\Work\Flood_Evacuation_System\3_outputs\flood_risk.tif"

with rasterio.open(flood_norm_path) as fsrc, rasterio.open(rain_norm_path) as rsrc:
    flood_norm = fsrc.read(1).astype("float32")
    rain_norm = rsrc.read(1).astype("float32")
    meta = fsrc.meta.copy()

# Weights
w_flood = 0.6
w_rain = 0.4

flood_risk = (w_flood * flood_norm) + (w_rain * rain_norm)

meta.update({
    "dtype": "float32",
    "count": 1
})

with rasterio.open(risk_out, "w", **meta) as dst:
    dst.write(flood_risk, 1)

print("Flood risk raster saved at:")
print(risk_out)
print("Risk min:", flood_risk.min())
print("Risk max:", flood_risk.max())

#  Classify flood risk into Low / Medium / High

risk_path = r"S:\Work\Flood_Evacuation_System\3_outputs\flood_risk.tif"
risk_class_out = r"S:\Work\Flood_Evacuation_System\3_outputs\flood_risk_classified.tif"

with rasterio.open(risk_path) as src:
    risk = src.read(1)
    meta = src.meta.copy()

# Classification
risk_class = np.zeros_like(risk, dtype="uint8")

risk_class[(risk >= 0.0) & (risk <= 0.33)] = 1   # Low
risk_class[(risk > 0.33) & (risk <= 0.66)] = 2   # Medium
risk_class[risk > 0.66] = 3                      # High

meta.update({
    "dtype": "uint8",
    "count": 1
})

with rasterio.open(risk_class_out, "w", **meta) as dst:
    dst.write(risk_class, 1)

print("Classified flood risk raster saved at:")
print(risk_class_out)
print("Unique risk classes:", sorted(set(risk_class.flatten())))


extent["north"] = extent["maxy"]
extent["south"] = extent["miny"]
extent["east"] = extent["maxx"]
extent["west"] = extent["minx"]



import json

extent_path = r"S:\Work\Flood_Evacuation_System\3_outputs\study_extent.json"

with open(extent_path, "w") as f:
    json.dump(extent, f)

print("Study extent saved at:", extent_path)

