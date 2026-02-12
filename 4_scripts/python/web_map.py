# STEP 36: Create interactive evacuation map using Folium

import folium
import geopandas as gpd

# Load evacuation route (GeoJSON)
route_path = r"S:\Work\Flood_Evacuation_System\3_outputs\evacuation_route.geojson"
route_gdf = gpd.read_file(route_path)

# Get map center
center_lat = route_gdf.geometry.iloc[0].centroid.y
center_lon = route_gdf.geometry.iloc[0].centroid.x

# Create Folium map
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=11,
    tiles="OpenStreetMap"
)

# Add evacuation route
folium.GeoJson(
    route_gdf,
    name="Evacuation Route",
    style_function=lambda x: {
        "color": "red",
        "weight": 5
    }
).add_to(m)

folium.LayerControl().add_to(m)

# Save map
map_path = r"S:\Work\Flood_Evacuation_System\3_outputs\evacuation_map.html"
m.save(map_path)

print("Evacuation web map saved at:")
print(map_path)
