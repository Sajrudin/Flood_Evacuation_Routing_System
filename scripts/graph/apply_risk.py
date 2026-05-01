import osmnx as ox
import networkx as nx
import joblib
import pandas as pd
from pathlib import Path


# Setup paths

BASE_DIR = Path(__file__).resolve().parents[2]

GRAPH_DIR = BASE_DIR / "outputs" / "graph"
MODEL_DIR = BASE_DIR / "models"

graph_utm_path = GRAPH_DIR / "roads_utm.graphml"
output_graph_path = GRAPH_DIR / "roads_ml.graphml"

model_path = MODEL_DIR / "best_flood_model.pkl"


# Load graph + model

print("Loading graph and ML model...")

G = ox.load_graphml(str(graph_utm_path))
largest_cc = max(nx.connected_components(G.to_undirected()), key=len)
G = G.subgraph(largest_cc).copy()

print("Graph cleaned (connected)")

model = joblib.load(str(model_path))

HAS_PROBA = hasattr(model, "predict_proba")

print("Graph loaded")
print("Model loaded")


# Apply ML risk to edges

print("Applying risk scores...")

total_edges = G.number_of_edges()

for i, (u, v, k, data) in enumerate(G.edges(keys=True, data=True)):

    if i % 5000 == 0:
        print(f"{i}/{total_edges} edges processed")

    base_length = float(data.get("length", 1.0))

    # ── Basic feature set (can improve later) ──
    features = pd.DataFrame([{
        "rainfall": 30,
        "dist_river": 200,
        "landuse": 1,
        "building_density": 5,
        "temp": 25,
        "humidity": 70
    }])

    # ── Predict risk ──
    try:
        if HAS_PROBA:
            risk_score = float(model.predict_proba(features)[0][1])
        else:
            risk_score = float(model.predict(features)[0])
    except Exception:
        risk_score = 0.0

    # Clamp
    risk_score = max(0.0, min(1.0, risk_score))

    # Store risk
    data["risk_score"] = risk_score

    # Default weight (balanced mode)
    data["weight"] = base_length * (1 + risk_score * 3)

    # Blocked roads (for UI only)
    data["blocked"] = risk_score > 0.8


# Save updated graph

ox.save_graphml(G, str(output_graph_path))

print(f"Graph with risk saved at: {output_graph_path}")
print("✅ Risk application complete")