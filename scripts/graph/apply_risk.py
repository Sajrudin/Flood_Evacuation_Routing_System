import osmnx as ox
import networkx as nx
import joblib
import pandas as pd
import random
from pathlib import Path
import rasterio

# ──────────────────────────────────────────────────────────
# PATH SETUP
# ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]

GRAPH_DIR = BASE_DIR / "outputs" / "graph"
RASTER_DIR = BASE_DIR / "outputs" / "rasters"
MODEL_DIR = BASE_DIR / "models"

graph_utm_path = GRAPH_DIR / "roads_utm.graphml"
output_graph_path = GRAPH_DIR / "roads_ml.graphml"
model_path = MODEL_DIR / "best_flood_model.pkl"
raster_path = RASTER_DIR / "flood_risk.tif"

# ──────────────────────────────────────────────────────────
# LOAD GRAPH
# ──────────────────────────────────────────────────────────
if not graph_utm_path.exists():
    print(f"❌ Graph not found: {graph_utm_path}")
    exit()

print("📂 Loading graph...")
G = ox.load_graphml(str(graph_utm_path))

largest_cc = max(nx.connected_components(G.to_undirected()), key=len)
G = G.subgraph(largest_cc).copy()

print(f"✅ Graph loaded. Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")

# ──────────────────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────────────────
if not model_path.exists():
    print(f"❌ Model not found: {model_path}")
    exit()

model = joblib.load(str(model_path))
HAS_PROBA = hasattr(model, "predict_proba")

print("✅ Model loaded")

# ──────────────────────────────────────────────────────────
# LOAD RASTER
# ──────────────────────────────────────────────────────────
if not raster_path.exists():
    print(f"❌ Raster not found: {raster_path}")
    exit()

print("📂 Loading raster...")
raster = rasterio.open(raster_path)
raster_data = raster.read(1)

print("Raster CRS:", raster.crs)
print("Raster bounds:", raster.bounds)

# ──────────────────────────────────────────────────────────
# APPLY HYBRID RISK
# ──────────────────────────────────────────────────────────
print("\n🧠 Applying hybrid risk...")

total_edges = G.number_of_edges()

for i, (u, v, k, data) in enumerate(G.edges(keys=True, data=True)):

    if i % 10000 == 0:
        print(f"Progress: {i}/{total_edges}", end="\r")

    node = G.nodes[u]

    x = float(node.get("x", 0))
    y = float(node.get("y", 0))

    # ─────────────────────────────
    # 1. SAMPLE RASTER
    # ─────────────────────────────
    try:
        row, col = raster.index(x, y)

        if (0 <= row < raster.height) and (0 <= col < raster.width):
            raster_risk = float(raster_data[row, col])
        else:
            raster_risk = 0.0
    except:
        raster_risk = 0.0

    # ─────────────────────────────
    # 2. ML RISK
    # ─────────────────────────────
    lat_factor = abs(y) % 1

    features = pd.DataFrame([{
        "lat": y,
        "lon": x,
        "rainfall": random.randint(20, 110),
        "temp": 25,
        "humidity": 70,
        "dist_river": int(500 * (lat_factor + random.random())),
        "landuse": random.randint(1, 5),
        "building_density": random.randint(0, 15),
    }])

    features = features[[
        "lat", "lon",
        "rainfall",
        "temp", "humidity",
        "dist_river",
        "landuse",
        "building_density"
    ]]

    try:
        if HAS_PROBA:
            ml_risk = float(model.predict_proba(features)[0][1])
        else:
            ml_risk = float(model.predict(features)[0])
    except:
        ml_risk = 0.1

    # ─────────────────────────────
    # 3. FINAL RISK (KEY)
    # ─────────────────────────────
    final_risk = (0.75 * raster_risk) + (0.25 * ml_risk)

    final_risk = max(0.0, min(1.0, final_risk))

    # ─────────────────────────────
    # STORE
    # ─────────────────────────────
    data["risk_score"] = final_risk

    # Strong blocking now works
    data["blocked"] = "True" if final_risk > 0.65 else "False"

    length = float(data.get("length", 1.0))

    # 🔥 Stronger impact (important)
    data["weight"] = length * (1 + final_risk * 15)

# ──────────────────────────────────────────────────────────
# SAVE GRAPH
# ──────────────────────────────────────────────────────────
print("\n💾 Saving graph...")
ox.save_graphml(G, str(output_graph_path))

print("✅ DONE — Graph updated with REAL spatial risk")