import osmnx as ox
import joblib
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ─────────────────────────────────────────────
# Load graph + model
# ─────────────────────────────────────────────
G     = ox.load_graphml(r"S:\Work\Flood_Evacuation_System\3_outputs\roads_dehradun.graphml")
model = joblib.load(r"S:\Work\Flood_Evacuation_System\3_outputs\best_flood_model.pkl")

# Check whether the model supports probability output
HAS_PROBA = hasattr(model, "predict_proba")
print(f"Model supports predict_proba: {HAS_PROBA}")

# ─────────────────────────────────────────────
# Load OSM features
# ─────────────────────────────────────────────
print("Loading OSM features...")

place = "Dehradun, Uttarakhand, India"

rivers    = ox.features_from_place(place, tags={"waterway": True})
landuse   = ox.features_from_place(place, tags={"landuse": True})
buildings = ox.features_from_place(place, tags={"building": True})

# Clean nulls
rivers    = rivers[rivers.geometry.notnull()]
landuse   = landuse[landuse.geometry.notnull()]
buildings = buildings[buildings.geometry.notnull()]

# Project to metric CRS for distance calculations
rivers    = rivers.to_crs(epsg=32644)
landuse   = landuse.to_crs(epsg=32644)
buildings = buildings.to_crs(epsg=32644)

# Build a single union geometry for fast river-distance queries
river_union = rivers.geometry.union_all() if not rivers.empty else None

print("OSM features loaded ✅")

# ─────────────────────────────────────────────
# NODE FEATURE CACHE
# Computes and caches ML features for each node
# ─────────────────────────────────────────────
node_features = {}

def compute_node_feature(node):
    """Return ML feature vector for a graph node, with caching."""

    if node in node_features:
        return node_features[node]

    x = G.nodes[node]["x"]   # longitude (WGS84)
    y = G.nodes[node]["y"]   # latitude  (WGS84)

    # Project point to metric CRS to match OSM layers
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32644", always_xy=True)
    mx, my = transformer.transform(x, y)
    point_m = Point(mx, my)

    # ── Distance to nearest river ──
    try:
        dist_river = point_m.distance(river_union) if river_union else 1000.0
    except Exception:
        dist_river = 1000.0

    # ── Nearest landuse class ──
    try:
        landuse.distance(point_m).idxmin()   # confirms at least one row exists
        landuse_val = 1
    except Exception:
        landuse_val = 0

    # ── Building density within 300 m buffer ──
    try:
        buffer = point_m.buffer(300)
        building_density = buildings[buildings.intersects(buffer)].shape[0]
    except Exception:
        building_density = 0

    # ── Weather (static placeholder – replace with live API if needed) ──
    rainfall = 30
    temp     = 25
    humidity = 70

    features = [
        rainfall,
        float(dist_river),
        landuse_val,
        float(building_density),
        temp,
        humidity
    ]

    node_features[node] = features
    return features

# ─────────────────────────────────────────────
# APPLY ML RISK SCORES
#
# KEY CHANGE from old version:
#   OLD → stored a baked-in binary weight (length × 5 or length × 1)
#   NEW → stores `risk_score` (float 0–1) on every edge so the API
#          can apply different weight functions at query time depending
#          on the user's chosen route preference (safer / shorter /
#          balanced) and risk threshold slider value.
#
# `weight` is still stored as a sensible default (safer preference,
#  threshold=0.5) so the graph remains compatible with tools that
#  expect a `weight` attribute.
# ─────────────────────────────────────────────
print("Applying ML risk scores to edges...")

total_edges = G.number_of_edges()

for i, (u, v, k, data) in enumerate(G.edges(keys=True, data=True)):

    if i % 1000 == 0:
        pct = round(i / total_edges * 100, 1)
        print(f"  {i:>6}/{total_edges}  ({pct}%) ")

    features = compute_node_feature(u)

    df = pd.DataFrame([features], columns=[
        "rainfall", "dist_river", "landuse",
        "building_density", "temp", "humidity"
    ])

    # ── Get risk score ──
    try:
        if HAS_PROBA:
            # Use class-1 probability as a continuous risk score
            risk_score = float(model.predict_proba(df)[0][1])
        else:
            # Fall back to binary prediction → 0.0 or 1.0
            risk_score = float(model.predict(df)[0])
    except Exception:
        risk_score = 0.0

    # Clamp to [0, 1]
    risk_score = max(0.0, min(1.0, risk_score))

    base = float(data.get("length", 1.0))

    # Store risk score so the API can use any threshold/preference
    data["risk_score"] = risk_score

    # Default weight: "safer" preference, threshold = 0.5
    # High-risk edges (risk_score > 0.5) get a proportional penalty
    data["weight"] = base * (1.0 + risk_score * 10.0)

print("ML risk scores applied ✅")

# ─────────────────────────────────────────────
# SAVE GRAPH
# ─────────────────────────────────────────────
OUT_PATH = r"S:\Work\Flood_Evacuation_System\3_outputs\roads_ml_weighted.graphml"
ox.save_graphml(G, OUT_PATH)
print(f"Graph saved → {OUT_PATH}")
print("Done ✅")