from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import osmnx as ox
import networkx as nx
import traceback

# ──────────────────────────────────────────────────────────
# PATH SETUP
# ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]

FRONTEND_DIR = BASE_DIR / "frontend"
GRAPH_PATH   = BASE_DIR / "outputs" / "graph" / "roads_ml.graphml"

# ──────────────────────────────────────────────────────────
# APP SETUP
# ──────────────────────────────────────────────────────────
app = FastAPI()

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────
# LOAD GRAPH
# ──────────────────────────────────────────────────────────
if GRAPH_PATH.exists():
    print("📂 Loading graph...")
    G = ox.load_graphml(str(GRAPH_PATH))

    if G.graph.get("crs") != "epsg:4326":
        print("Transforming graph to EPSG:4326...")
        G = ox.project_graph(G, to_crs="EPSG:4326")
else:
    raise FileNotFoundError("Graph not found. Run apply_risk.py first.")

G = ox.convert.to_undirected(G)

# ──────────────────────────────────────────────────────────
# NORMALIZE ATTRIBUTES
# ──────────────────────────────────────────────────────────
for u, v, data in G.edges(data=True):
    data["length"]     = float(data.get("length", 1.0))
    data["risk_score"] = float(data.get("risk_score", 0.0))
    data["blocked"]    = str(data.get("blocked", "False")) == "True"

print("✅ Graph loaded successfully")
print("Nodes:", G.number_of_nodes(), "| Edges:", G.number_of_edges())

# Debug risk stats
risks = [data["risk_score"] for _, _, data in G.edges(data=True)]
print("\n===== RISK STATS =====")
print("Min:", min(risks))
print("Max:", max(risks))
print("Avg:", sum(risks) / len(risks))

G = nx.freeze(G)

# ──────────────────────────────────────────────────────────
# EDGE WEIGHT FUNCTION
# ──────────────────────────────────────────────────────────
def edge_weight_modified(u, v, d, mode="balanced"):
    if isinstance(d, dict) and 0 in d:
        d = d[0]

    length  = d.get("length", 1.0)
    risk    = d.get("risk_score", 0.0)
    blocked = d.get("blocked", False)

    # Strong scaling
    scaled_risk = risk * 10

    if mode == "shortest":
        return length

    elif mode == "safe":
        if blocked:
            return length * 1000  
        return length * (1 + scaled_risk * 20)

    elif mode == "balanced":
        return length * (1 + scaled_risk * 5)

    elif mode == "unsafe":
        return max(length * (1 - scaled_risk * 0.9), 0.01)

    return length


# ──────────────────────────────────────────────────────────
# MULTI ROUTE FUNCTION  (✅ FIXED: unpack (cost, path) tuple)
# ──────────────────────────────────────────────────────────
def compute_routes(G, src_node, dst_node):
    routes = {}
    modes  = ["shortest", "safe", "balanced", "unsafe"]

    for mode in modes:
        try:
            _, path = nx.single_source_dijkstra(
                G, src_node, dst_node,
                weight=lambda u, v, d, m=mode: edge_weight_modified(u, v, d, m)
            )
            routes[mode] = path
            print(f"[{mode}] path length: {len(path)} nodes")

        except nx.NetworkXNoPath:
            print(f"[{mode}] No path found")
        except Exception as e:
            print(f"[{mode}] Error: {e}")

    return routes if routes else None


# ──────────────────────────────────────────────────────────
# ROUTE → GEOJSON
# ──────────────────────────────────────────────────────────
def route_to_geojson(G, route):
    features = []

    for i in range(len(route) - 1):
        u = route[i]
        v = route[i + 1]

        edge_data = G.get_edge_data(u, v)
        if not edge_data:
            print(f"⚠️  Missing edge data for ({u}, {v})")
            continue

        # MultiGraph: pick first parallel edge
        edge = list(edge_data.values())[0]

        lon1, lat1 = G.nodes[u]["x"], G.nodes[u]["y"]
        lon2, lat2 = G.nodes[v]["x"], G.nodes[v]["y"]

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[lon1, lat1], [lon2, lat2]]
            },
            "properties": {
                "risk_score": round(edge.get("risk_score", 0.0), 3),
                "blocked":    edge.get("blocked", False),
                "length": edge.get("length", 0.0)
            }
        })

    return {
        "type": "FeatureCollection",
        "features": features
    }


# ──────────────────────────────────────────────────────────
# ROUTES API
# ──────────────────────────────────────────────────────────
@app.get("/")
def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "nodes":  G.number_of_nodes(),
        "edges":  G.number_of_edges()
    }


@app.post("/route")
async def route(request: Request):
    body = await request.json()

    src_lat, src_lon = body["source"]
    dst_lat, dst_lon = body["destination"]

    try:
        src_node = ox.distance.nearest_nodes(G, X=src_lon, Y=src_lat)
        dst_node = ox.distance.nearest_nodes(G, X=dst_lon, Y=dst_lat)
    except Exception as e:
        print(f"nearest_nodes error: {e}")
        return {"error": "Invalid coordinates"}

    print(f"Routing from node {src_node} → {dst_node}")

    if not nx.has_path(G, src_node, dst_node):
        return {"error": "No path exists between these points"}

    routes = compute_routes(G, src_node, dst_node)

    if routes is None:
        return {"error": "No route found"}

    return {
        key: route_to_geojson(G, path)
        for key, path in routes.items()
    }