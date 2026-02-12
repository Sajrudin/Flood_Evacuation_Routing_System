from flask import Flask, request, jsonify
from flask_cors import CORS
import osmnx as ox
import networkx as nx
from pyproj import Transformer
from shapely.geometry import LineString

app = Flask(__name__)
CORS(app)

# Load SAFE graph once
GRAPH_PATH = r"S:\Work\Flood_Evacuation_System\3_outputs\roads_dehradun_safe.graphml"
G = ox.load_graphml(GRAPH_PATH)

# CRS transformers
to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32644", always_xy=True)
to_wgs = Transformer.from_crs("EPSG:32644", "EPSG:4326", always_xy=True)

@app.route("/route", methods=["POST"])
def route():
    data = request.json
    src = data["source"]
    dst = data["destination"]

    # Convert lat/lon → UTM
    sx, sy = to_utm.transform(src[1], src[0])
    dx, dy = to_utm.transform(dst[1], dst[0])

    src_node = ox.distance.nearest_nodes(G, sx, sy)
    dst_node = ox.distance.nearest_nodes(G, dx, dy)

    path = nx.shortest_path(G, src_node, dst_node, weight="length")
    length = nx.shortest_path_length(G, src_node, dst_node, weight="length")

    # Build route geometry
    coords_utm = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in path]
    coords_ll = [to_wgs.transform(x, y) for x, y in coords_utm]


    # Simple avoided estimate (demo-safe)
    avoided_segments = int(length / 1000 * 0.5)  # ~0.5 unsafe/km (demo logic)

    return jsonify({
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": coords_ll
        },
        "properties": {
            "distance_km": round(length / 1000, 2),
            "avoided_unsafe": avoided_segments
        }
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)