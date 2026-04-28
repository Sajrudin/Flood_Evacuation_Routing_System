from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import osmnx as ox
import networkx as nx

# ─────────────────────────────────────────────
# PATH SETUP (IMPORTANT: define first)
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]

FRONTEND_DIR = BASE_DIR / "frontend"
GRAPH_PATH   = BASE_DIR / "outputs" / "graph" / "roads_weighted.graphml"

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = FastAPI()

# Serve frontend (CSS, JS)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Enable CORS (safe for development)
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
print("⏳ Loading graph...")

G = ox.load_graphml(str(GRAPH_PATH))
G = G.to_undirected()

print("Graph is directed:", G.is_directed())

# Normalize edge attributes
for u, v, k, data in G.edges(keys=True, data=True):
    try:
        data["weight"]     = float(data.get("weight", 1.0))
        data["length"]     = float(data.get("length", 1.0))
        data["risk_score"] = float(data.get("risk_score", 0.0))
    except Exception:
        data["weight"]     = 1.0
        data["length"]     = 1.0
        data["risk_score"] = 0.0

print("✅ Graph loaded successfully")

# ─────────────────────────────────────────────
# SPEED SETTINGS
# ─────────────────────────────────────────────
SPEED_SAFE   = 35.0
SPEED_UNSAFE = 12.0

# ─────────────────────────────────────────────
# DYNAMIC WEIGHT FUNCTION
# ─────────────────────────────────────────────
def make_weight_fn(preference: str, risk_threshold: float):

    def weight_fn(u, v, data: dict) -> float:
        length     = float(data.get("length", 1.0))
        risk_score = float(data.get("risk_score", 0.0))

        if preference == "shorter":
            return length

        elif preference == "safer":
            excess = max(0.0, risk_score - risk_threshold)
            multiplier = 1.0 + (excess / (1.0 - risk_threshold + 1e-6)) ** 2 * 20.0
            return length * multiplier

        else:  # balanced
            excess = max(0.0, risk_score - risk_threshold)
            multiplier = 1.0 + (excess / (1.0 - risk_threshold + 1e-6)) * 5.0
            return length * multiplier

    return weight_fn

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

# Serve frontend
@app.get("/")
def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")

# Health check
@app.get("/health")
def health():
    return {
        "status": "ok",
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges()
    }

# Route API
@app.post("/route")
async def route(request: Request):
    body = await request.json()

    # Required inputs
    src_coords = body["source"]
    dst_coords = body["destination"]

    # Optional inputs
    risk_threshold = float(body.get("risk_threshold", 0.65))
    preference     = str(body.get("route_preference", "safer")).lower()
    
    if preference not in ("safer", "shorter", "balanced"):
        preference = "safer"

    risk_threshold = max(0.0, min(1.0, risk_threshold))

    # Snap coordinates
    try:
        src_node = ox.distance.nearest_nodes(G, src_coords[1], src_coords[0])
        dst_node = ox.distance.nearest_nodes(G, dst_coords[1], dst_coords[0])
    except Exception:
        return {"error": "Invalid coordinates"}

    print("Source node:", src_node)
    print("Destination node:", dst_node)
    print("Neighbors of source:", list(G.neighbors(src_node))[:5])
    # Compute route
    weight_fn = make_weight_fn(preference, risk_threshold)

    try:
        path = nx.shortest_path(G, src_node, dst_node, weight=weight_fn)
    except nx.NetworkXNoPath:
        return {"error": "No route found"}
    except nx.NodeNotFound:
        return {"error": "Point outside road network"}

    # Build response
    segments        = []
    total_length_m  = 0.0
    travel_time_sec = 0.0
    unsafe_count    = 0

    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]

        edge_data = G.get_edge_data(u, v)
        edge      = list(edge_data.values())[0]

        length     = float(edge.get("length", 1.0))
        risk_score = float(edge.get("risk_score", 0.0))

        total_length_m += length

        is_unsafe = risk_score > risk_threshold

        if is_unsafe:
            unsafe_count += 1
            travel_time_sec += (length / 1000) / SPEED_UNSAFE * 3600
        else:
            travel_time_sec += (length / 1000) / SPEED_SAFE * 3600

        lon1, lat1 = G.nodes[u]["x"], G.nodes[u]["y"]
        lon2, lat2 = G.nodes[v]["x"], G.nodes[v]["y"]

        segments.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[lon1, lat1], [lon2, lat2]]
            },
            "properties": {
                "unsafe": is_unsafe,
                "risk_score": round(risk_score, 3),
                "length_m": round(length, 1)
            }
        })

    return {
        "type": "FeatureCollection",
        "features": segments,
        "properties": {
            "distance_km": round(total_length_m / 1000, 2),
            "travel_time_min": round(travel_time_sec / 60, 1),
            "avoided_unsafe": unsafe_count,
            "total_segments": len(segments),
            "risk_threshold": risk_threshold,
            "route_preference": preference
        }
    }