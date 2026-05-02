import os
os.environ["SHAPE_RESTORE_SHX"] = "YES"

import numpy as np
import geopandas as gpd
from pathlib import Path
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from scipy.spatial import cKDTree
import json

# ──────────────────────────────────────────────────────────
# PATH SETUP (RELATIVE)
# ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]

RAW_RASTER_DIR = BASE_DIR / "data" / "raw" / "flood_rasters"
RAW_DWA_DIR = BASE_DIR / "data" / "raw" / "waterbodies"
OUTPUT_DIR = BASE_DIR / "outputs" / "rasters"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

imd_path = RAW_DWA_DIR / "IMD_Station.shp"

flood_files = {
    "2008": RAW_RASTER_DIR / "flood_2008.tif",
    "2009": RAW_RASTER_DIR / "flood_2009.tif",
    "2010": RAW_RASTER_DIR / "flood_2010.tif",
}

# ──────────────────────────────────────────────────────────
# LOAD IMD DATA
# ──────────────────────────────────────────────────────────
imd = gpd.read_file(imd_path)
imd_utm = imd.to_crs(epsg=32644)

imd_utm = imd_utm[imd_utm["RF_1Day"].notnull()]
imd_utm = imd_utm[imd_utm["RF_1Day"] > 0]

# ──────────────────────────────────────────────────────────
# CREATE EXTENT
# ──────────────────────────────────────────────────────────
minx, miny, maxx, maxy = imd_utm.total_bounds
buffer = 10000

extent = {
    "minx": minx - buffer,
    "miny": miny - buffer,
    "maxx": maxx + buffer,
    "maxy": maxy + buffer
}

# 🔥 FIX 1: Better resolution
resolution = 250  # MUCH better than 1000

x_coords = np.arange(extent["minx"], extent["maxx"], resolution)
y_coords = np.arange(extent["miny"], extent["maxy"], resolution)

xx, yy = np.meshgrid(x_coords, y_coords)
grid_points = np.column_stack([xx.ravel(), yy.ravel()])

# ──────────────────────────────────────────────────────────
# IDW INTERPOLATION
# ──────────────────────────────────────────────────────────
station_coords = np.column_stack([
    imd_utm.geometry.x.values,
    imd_utm.geometry.y.values
])

rainfall_values = imd_utm["RF_1Day"].values

def idw_chunked(station_xy, values, grid_xy, power=2, k=8, chunk_size=50000):
    tree = cKDTree(station_xy)
    result = np.empty(len(grid_xy), dtype="float32")

    for i in range(0, len(grid_xy), chunk_size):
        chunk = grid_xy[i:i + chunk_size]

        dist, idx = tree.query(chunk, k=k)

        weights = 1 / (dist ** power)
        weights[dist == 0] = 1e12

        weighted_vals = np.sum(weights * values[idx], axis=1)
        result[i:i + len(chunk)] = weighted_vals / np.sum(weights, axis=1)

        print(f"Processed {i + len(chunk)} / {len(grid_xy)}", end="\r")

    return result

print("🌧️ Generating rainfall surface...")
rainfall_idw = idw_chunked(station_coords, rainfall_values, grid_points)

rainfall_raster = rainfall_idw.reshape(len(y_coords), len(x_coords))

transform = from_origin(extent["minx"], extent["maxy"], resolution, resolution)

rainfall_path = OUTPUT_DIR / "rainfall_idw.tif"

with rasterio.open(
    rainfall_path, "w",
    driver="GTiff",
    height=rainfall_raster.shape[0],
    width=rainfall_raster.shape[1],
    count=1,
    dtype="float32",
    crs="EPSG:32644",
    transform=transform
) as dst:
    dst.write(rainfall_raster.astype("float32"), 1)

# ──────────────────────────────────────────────────────────
# REPROJECT FLOOD RASTERS
# ──────────────────────────────────────────────────────────
reprojected = []

for year, path in flood_files.items():
    out_path = OUTPUT_DIR / f"flood_{year}_utm.tif"

    with rasterio.open(path) as src:
        dest = np.zeros_like(rainfall_raster, dtype="uint8")

        reproject(
            source=src.read(1),
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs="EPSG:32644",
            resampling=Resampling.nearest
        )

    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=dest.shape[0],
        width=dest.shape[1],
        count=1,
        dtype="uint8",
        crs="EPSG:32644",
        transform=transform
    ) as dst:
        dst.write(dest, 1)

    reprojected.append(out_path)

# ──────────────────────────────────────────────────────────
# FLOOD FREQUENCY
# ──────────────────────────────────────────────────────────
freq = sum([rasterio.open(p).read(1) for p in reprojected])
freq = freq / 3.0  # normalize

# ──────────────────────────────────────────────────────────
# NORMALIZE RAINFALL
# ──────────────────────────────────────────────────────────
rain = rainfall_raster
rain_norm = (rain - rain.min()) / (rain.max() - rain.min())

# 🔥 FIX 2: Contrast stretch (VERY IMPORTANT)
rain_norm = np.clip(rain_norm * 1.5, 0, 1)

# ──────────────────────────────────────────────────────────
# FINAL RISK MAP
# ──────────────────────────────────────────────────────────

# 🔥 FIX 3: Stronger flood importance
flood_risk = (0.7 * freq) + (0.3 * rain_norm)

# 🔥 FIX 4: Boost high-risk zones
flood_risk = np.power(flood_risk, 1.5)

risk_path = OUTPUT_DIR / "flood_risk.tif"

with rasterio.open(
    risk_path, "w",
    driver="GTiff",
    height=flood_risk.shape[0],
    width=flood_risk.shape[1],
    count=1,
    dtype="float32",
    crs="EPSG:32644",
    transform=transform
) as dst:
    dst.write(flood_risk.astype("float32"), 1)

print("\n✅ FINAL RASTER GENERATED")
print("Min:", flood_risk.min())
print("Max:", flood_risk.max())

# ──────────────────────────────────────────────────────────
# SAVE EXTENT
# ──────────────────────────────────────────────────────────
extent.update({
    "north": extent["maxy"],
    "south": extent["miny"],
    "east": extent["maxx"],
    "west": extent["minx"]
})

with open(OUTPUT_DIR / "study_extent.json", "w") as f:
    json.dump(extent, f)

print("📁 Outputs saved in:", OUTPUT_DIR)