import osmnx as ox
import networkx as nx
from pathlib import Path


# Setup project paths (ROOT → outputs/graph)

BASE_DIR = Path(__file__).resolve().parents[2]

GRAPH_DIR = BASE_DIR / "outputs" / "graph"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

graph_path = GRAPH_DIR / "roads.graphml"
graph_utm_path = GRAPH_DIR / "roads_utm.graphml"


# Download road network

print("Downloading expanded road network...")

place = "Dehradun, Uttarakhand, India"

G = ox.graph_from_place(
    place,
    network_type="drive",
    simplify=False,
    retain_all=True
)

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())


# Keep largest connected component

largest_cc = max(nx.connected_components(G.to_undirected()), key=len)
G = G.subgraph(largest_cc).copy()


# Save original graph

ox.save_graphml(G, str(graph_path)) # type: ignore
print(f"Graph saved at: {graph_path}")


# Project graph to UTM (for distance calculations)

G = ox.project_graph(G, to_crs="EPSG:32644")

ox.save_graphml(G, str(graph_utm_path))
print(f"Projected graph saved at: {graph_utm_path}")

print("Download + projection complete")