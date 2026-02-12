# # # # STEP 22 FIX: Convert study extent to lat/lon for OSMnx
# # # import json

# # # extent_path = r"S:\Work\Flood_Evacuation_System\3_outputs\study_extent.json"

# # # with open(extent_path, "r") as f:
# # #     extent = json.load(f)

# # # print("Loaded extent:", extent)

# # # # STEP 25: Download road network using place-based query (RECOMMENDED)

# # # import osmnx as ox

# # # place_name = "Dehradun district, Uttarakhand, India"

# # # print("Downloading road network from OSM using place name...")

# # # G = ox.graph_from_place(
# # #     place_name,
# # #     network_type="drive",
# # #     simplify=True
# # # )

# # # print("Road network downloaded successfully.")
# # # print("Number of nodes:", G.number_of_nodes())
# # # print("Number of edges:", G.number_of_edges())

# # # ox.save_graphml(
# # #     G,
# # #     r"S:\Work\Flood_Evacuation_System\3_outputs\roads_dehradun.graphml"
# # # )

# # # print("Road network saved.")

# # # STEP 26: Project road network to EPSG:32644

import osmnx as ox

graph_path = r"S:\Work\Flood_Evacuation_System\3_outputs\roads_dehradun.graphml"

# # # Load saved graph (to avoid re-downloading)
G = ox.load_graphml(graph_path)

# # # Project to UTM 44N (EPSG:32644)
G_proj = ox.project_graph(G, to_crs="EPSG:32644")

print("Road network projected to EPSG:32644.")

# # # Save projected graph
proj_graph_path = r"S:\Work\Flood_Evacuation_System\3_outputs\roads_dehradun_utm.graphml"
ox.save_graphml(G_proj, proj_graph_path)

print("Projected road network saved at:")
print(proj_graph_path)

proj_graph_path = r"S:\Work\Flood_Evacuation_System\3_outputs\roads_dehradun_utm.graphml"

G_proj = ox.load_graphml(proj_graph_path)

# Extract edges GeoDataFrame
edges_gdf = ox.graph_to_gdfs(
    G_proj,
    nodes=False,
    edges=True
)

# # print("Edges GeoDataFrame created.")
# # print("Number of road segments:", len(edges_gdf))
# # print(edges_gdf.head())

# # STEP 28: Sample flood risk raster on road segments

import rasterio
import numpy as np

risk_raster_path = r"S:\Work\Flood_Evacuation_System\3_outputs\flood_risk.tif"

with rasterio.open(risk_raster_path) as src:
    risk_array = src.read(1)
    transform = src.transform

def sample_risk(geometry):
    # Use centroid of road segment
    x, y = geometry.centroid.x, geometry.centroid.y
    row, col = src.index(x, y)

    # Safety check
    if 0 <= row < risk_array.shape[0] and 0 <= col < risk_array.shape[1]:
        return risk_array[row, col]
    else:
        return np.nan

# # Apply sampling
edges_gdf["flood_risk"] = edges_gdf.geometry.apply(sample_risk)

# print("Flood risk sampled on road segments.")
# print(edges_gdf[["highway", "flood_risk"]].head())

# print("Flood risk stats on roads:")
# print(edges_gdf["flood_risk"].describe())

RISK_THRESHOLD = 0.12

edges_gdf["unsafe"] = edges_gdf["flood_risk"] >= RISK_THRESHOLD
# STEP B: Export unsafe roads for web map (GeoJSON)

unsafe_gdf = edges_gdf[edges_gdf["unsafe"]].copy()

# Convert to lat/lon for Leaflet
unsafe_gdf = unsafe_gdf.to_crs(epsg=4326)

unsafe_roads_path = r"S:\Work\Flood_Evacuation_System\3_outputs\unsafe_roads.geojson"

unsafe_gdf.to_file(unsafe_roads_path, driver="GeoJSON")

print("Unsafe roads GeoJSON saved at:")
print(unsafe_roads_path)


# print("Unsafe road threshold:", RISK_THRESHOLD)
# print("Total road segments:", len(edges_gdf))
# print("Unsafe road segments:", edges_gdf["unsafe"].sum())
# print("Safe road segments:", (~edges_gdf["unsafe"]).sum())

# print("\nSample unsafe roads:")
# print(edges_gdf[edges_gdf["unsafe"]][["highway", "flood_risk"]].head())


# # STEP 30: Remove unsafe road segments from the graph

# # STEP 30: Remove unsafe road segments and save safe graph (GraphML-safe)

# # STEP 30: Remove unsafe road segments and save safe graph (OSMnx-safe)

# import osmnx as ox

# # Work on a copy
# G_safe = G_proj.copy()

# removed_count = 0

# # Remove unsafe edges
# for (u, v, key), row in edges_gdf.iterrows():
#     if row["unsafe"]:
#         if G_safe.has_edge(u, v, key):
#             G_safe.remove_edge(u, v, key)
#             removed_count += 1

# print("Unsafe edges removed:", removed_count)

# print("Safe graph stats:")
# print("Nodes:", G_safe.number_of_nodes())
# print("Edges:", G_safe.number_of_edges())

# # Save using OSMnx (handles attribute cleaning internally)
# safe_graph_path = r"S:\Work\Flood_Evacuation_System\3_outputs\roads_dehradun_safe.graphml"

# ox.save_graphml(G_safe, safe_graph_path)

# print("Safe road network saved successfully at:")
# print(safe_graph_path)

# STEP 31: Select source and destination nodes

# STEP 31 FIXED: Select source & destination nodes (CRS-safe)

import osmnx as ox
from pyproj import Transformer

safe_graph_path = r"S:\Work\Flood_Evacuation_System\3_outputs\roads_dehradun_safe.graphml"

# Load safe road graph (already in EPSG:32644)
G_safe = ox.load_graphml(safe_graph_path)

# Lat/Lon coordinates
src_lat, src_lon = 30.3165, 78.0322     # Flood-prone area
dst_lat, dst_lon = 30.4590, 78.0650     # Safer high-ground area

# Transformer: WGS84 → UTM 44N
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32644", always_xy=True)

# Convert to UTM
src_x, src_y = transformer.transform(src_lon, src_lat)
dst_x, dst_y = transformer.transform(dst_lon, dst_lat)

# Find nearest nodes using UTM coordinates
source_node = ox.distance.nearest_nodes(G_safe, src_x, src_y)
dest_node = ox.distance.nearest_nodes(G_safe, dst_x, dst_y)

print("Source node:", source_node)
print("Destination node:", dest_node)


# STEP 32: Compute safest evacuation route (shortest path)

import networkx as nx

print("Computing safest evacuation route...")

# Shortest path based on length
route_nodes = nx.shortest_path(
    G_safe,
    source=source_node,
    target=dest_node,
    weight="length"
)

# Compute total route length (meters)
route_length = nx.shortest_path_length(
    G_safe,
    source=source_node,
    target=dest_node,
    weight="length"
)

print("Evacuation route computed successfully.")
print("Number of nodes in route:", len(route_nodes))
print("Total route length (meters):", round(route_length, 2))

# STEP 33: Convert evacuation route to geometry

import geopandas as gpd
from shapely.geometry import LineString

# Extract node coordinates (already in EPSG:32644)
route_coords = [
    (G_safe.nodes[node]["x"], G_safe.nodes[node]["y"])
    for node in route_nodes
]

# Create LineString
route_line = LineString(route_coords)

# Create GeoDataFrame
route_gdf = gpd.GeoDataFrame(
    {"type": ["evacuation_route"]},
    geometry=[route_line],
    crs="EPSG:32644"
)

print("Evacuation route geometry created.")
print(route_gdf)

# STEP 34: Visualize evacuation route over flood risk (static)

import matplotlib.pyplot as plt
import rasterio.plot

risk_raster_path = r"S:\Work\Flood_Evacuation_System\3_outputs\flood_risk.tif"

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# Plot flood risk raster
with rasterio.open(risk_raster_path) as src:
    rasterio.plot.show(
        src,
        ax=ax,
        cmap="Blues",
        title="Flood Risk with Safe Evacuation Route"
    )

# Plot evacuation route
route_gdf.plot(
    ax=ax,
    color="red",
    linewidth=2,
    label="Evacuation Route"
)

plt.legend()
plt.show()

# STEP 35: Export evacuation route for web map (GeoJSON)

from pyproj import Transformer

# Convert route to EPSG:4326 for web maps
route_web = route_gdf.to_crs(epsg=4326)

route_geojson_path = r"S:\Work\Flood_Evacuation_System\3_outputs\evacuation_route.geojson"

route_web.to_file(route_geojson_path, driver="GeoJSON")

print("Evacuation route exported for web map at:")
print(route_geojson_path)
