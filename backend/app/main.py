from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import osmnx as ox
import networkx as nx

# ─────────────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]

FRONTEND_DIR = BASE_DIR / "frontend"
GRAPH_PATH   = BASE_DIR / "outputs" / "graph" / "roads_ml.graphml"

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = FastAPI()

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# LOAD GRAPH
# ─────────────────────────────────────────────
if GRAPH_PATH.exists():
    print("📂 Loading graph...")
    G = ox.load_graphml(str(GRAPH_PATH))
else:
    raise FileNotFoundError("Graph not found. Run apply_risk.py first.")

# Convert to undirected for routing
G = G.to_undirected()
print("Sample edge:", list(G.edges(data=True))[0])


# Normalize attributes (important)
for u, v, data in G.edges(data=True):

    # length
    try:
        data["length"] = float(data.get("length", 1.0))
    except:
        data["length"] = 1.0

    # risk_score
    try:
        data["risk_score"] = float(data.get("risk_score", 0.0))
    except:
        data["risk_score"] = 0.0

    # blocked
    data["blocked"] = str(data.get("blocked", "False")) == "True"

print("✅ Graph loaded successfully")
print("Nodes:", G.number_of_nodes(), "| Edges:", G.number_of_edges())


# EDGE WEIGHT FUNCTION (FIXED)

def edge_weight(u, v, d, mode="balanced"):

    length = float(d.get("length", 1))
    risk = float(d.get("risk_score", 0))

    if mode == "shortest":
        w = length

    elif mode == "safe":
        w = length * (1 + risk * 5)

    elif mode == "balanced":
        w = length * (1 + risk * 2)

    elif mode == "unsafe":
        w = length * (1 - risk * 0.5)

    else:
        w = length

    return max(w, 0.1)


# MULTI ROUTE FUNCTION

def compute_routes(G, src_node, dst_node):

    routes = {}

    try:
        print("➡️ Trying SHORTEST")
        routes["shortest"] = nx.shortest_path(
            G, src_node, dst_node,
            weight=lambda u, v, d: edge_weight(u, v, d, "shortest")
        )
        print("✅ shortest OK")

    except Exception as e:
        print("❌ shortest FAILED:", e)

    try:
        print("➡️ Trying SAFE")
        routes["safe"] = nx.shortest_path(
            G, src_node, dst_node,
            weight=lambda u, v, d: edge_weight(u, v, d, "safe")
        )
        print("✅ safe OK")

    except Exception as e:
        print("❌ safe FAILED:", e)

    try:
        print("➡️ Trying BALANCED")
        routes["balanced"] = nx.shortest_path(
            G, src_node, dst_node,
            weight=lambda u, v, d: edge_weight(u, v, d, "balanced")
        )
        print("✅ balanced OK")

    except Exception as e:
        print("❌ balanced FAILED:", e)

    try:
        print("➡️ Trying UNSAFE")
        routes["unsafe"] = nx.shortest_path(
            G, src_node, dst_node,
            weight=lambda u, v, d: edge_weight(u, v, d, "unsafe")
        )
        print("✅ unsafe OK")

    except Exception as e:
        print("❌ unsafe FAILED:", e)

    if len(routes) == 0:
        return None

    return routes

# ─────────────────────────────────────────────
# CONVERT ROUTE → GEOJSON
# ─────────────────────────────────────────────
def route_to_geojson(G, route):

    features = []

    for i in range(len(route) - 1):
        u = route[i]
        v = route[i + 1]

        # 🔥 FIX: check both directions
        edge_data = G.get_edge_data(u, v) or G.get_edge_data(v, u)

        if edge_data is None:
            continue

        # handle dict structure
        if isinstance(edge_data, dict) and 0 in edge_data:
            edge = edge_data[0]
        else:
            edge = edge_data

        lon1, lat1 = G.nodes[u]["x"], G.nodes[u]["y"]
        lon2, lat2 = G.nodes[v]["x"], G.nodes[v]["y"]

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[lon1, lat1], [lon2, lat2]]
            },
            "properties": {
                "risk_score": round(float(edge.get("risk_score", 0)), 3),
                "blocked": edge.get("blocked", False)
            }
        })

    return {
        "type": "FeatureCollection",
        "features": features
    }

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/")
def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges()
    }

@app.post("/route")
async def route(request: Request):

    body = await request.json()

    src_lat, src_lon = body["source"]
    dst_lat, dst_lon = body["destination"]

    try:
        src_node = ox.distance.nearest_nodes(G, src_lon, src_lat)
        dst_node = ox.distance.nearest_nodes(G, dst_lon, dst_lat)
    except Exception:
        return {"error": "Invalid coordinates"}

    # Check connectivity
    if not nx.has_path(G, src_node, dst_node):
        print("❌ NO PATH BETWEEN NODES")
        return {"error": "No route found between selected points"}

    routes = compute_routes(G, src_node, dst_node)

    if routes is None:
        return {"error": "No route found"}

    response = {}

    for key, path in routes.items():
        response[key] = route_to_geojson(G, path)

    return response