# 🌊 Flood Evacuation Route Planner

A smart geospatial web application that computes **safe, shortest, and risk-aware evacuation routes** during flood scenarios using real-world road networks and risk modeling.

---

## 🚀 Project Overview

The **Flood Evacuation Route Planner** is designed to assist users in navigating through flood-prone areas by providing multiple route options based on:

* 📏 Distance (Shortest path)
* 🛡️ Safety (Avoiding high-risk zones)
* ⚖️ Balanced trade-off
* ⚠️ Risk-seeking (for analysis/testing)

The system integrates **geospatial data, graph algorithms, and real-time interaction** to deliver dynamic route visualization on an interactive map.

---

## 🧠 Key Features

### 🗺️ Interactive Map Interface

* Click or search to select **source and destination**
* Real-time route rendering using Leaflet
* Smooth zoom and pan functionality

### 🧭 Multiple Route Types

* 🔵 **Shortest** – Minimum distance
* 🟢 **Safe** – Avoids risky/blocked roads
* 🟠 **Balanced** – Trade-off between distance and safety
* 🔴 **Unsafe** – Prioritizes risky paths (for simulation)

### 📊 Dynamic Route Summary

* Total distance (km)
* Estimated time (based on realistic speed)
* Unsafe segments count

### 🖱️ Hover-Based Insights

* View **segment-level details**:

  * Distance
  * Risk score
  * Blocked status

### 📌 Route Interaction

* Toggle between route types
* Highlight routes on hover
* Popup with complete route summary

---

## 🏗️ System Architecture

```
Frontend (HTML/CSS/JS + Leaflet)
        ↓
FastAPI Backend (Python)
        ↓
OSMnx + NetworkX Graph
        ↓
GeoJSON Response → Map Rendering
```

---

## ⚙️ Tech Stack

### Frontend

* HTML5, CSS3, JavaScript
* Leaflet.js (map rendering)
* Leaflet Control Geocoder

### Backend

* FastAPI (Python web framework)
* OSMnx (road network extraction)
* NetworkX (graph algorithms)

### Data

* OpenStreetMap (road network)
* Custom risk scoring model

---

## 🧮 Routing Logic

Routes are computed using **graph-based shortest path algorithms**:

* Uses **Dijkstra / A*** with custom edge weights
* Edge weight depends on:

  * Road length
  * Risk score
  * Blocked status

### Risk-Based Weighting

| Mode     | Strategy                    |
| -------- | --------------------------- |
| Shortest | Distance only               |
| Safe     | High penalty on risky edges |
| Balanced | Moderate penalty            |
| Unsafe   | Prefers high-risk edges     |

---

## 📂 Project Structure

```
Flood_Evacuation_System/
│
├── backend/
│   └── app/
│       └── main.py
│
├── frontend/
│   ├── index.html
│   ├── css/
│   │   └── styles.css
│   └── js/
│       └── app.js
│
├── outputs/
│   └── graph/
│       └── roads_ml.graphml
│
└── README.md
```

---

## ▶️ How to Run

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/flood-evacuation-planner.git
cd flood-evacuation-planner
```

### 2️⃣ Install Dependencies

```bash
pip install fastapi uvicorn osmnx networkx
```

### 3️⃣ Run Backend

```bash
uvicorn backend.app.main:app --reload
```

### 4️⃣ Open Application

Visit:

```
http://127.0.0.1:8000
```

---

## 📈 Performance Considerations

* Graph size: ~300k+ nodes
* Uses optimized routing strategies
* Future improvements:

  * A* algorithm for faster routing
  * Graph pruning (region-based)
  * Caching frequent routes

---

## ⚠️ Limitations

* Routes may overlap in regions with limited road options
* Risk scores depend on available data (can be enhanced)
* Performance may vary with large graphs

---

## 🔮 Future Enhancements

* 🌧️ Real-time flood data integration
* 🧠 Machine learning-based risk prediction
* 📍 GPS/live tracking
* 📊 Risk heatmap overlay
* ⚡ Faster routing (A* + caching)

---

## 👨‍💻 Author

**Sajrudin Aalam**
B.Tech CSE | Graphic Era Hill University

* 📍 Dehradun, India
* 🔗 [LinkedIn](https://www.linkedin.com/in/sajrudin-aalam-21b861287/)

---

## 📜 License

This project is for educational and research purposes.

---

## ⭐ Acknowledgements

* OpenStreetMap for geospatial data
* OSMnx & NetworkX for graph processing
* Leaflet.js for interactive maps

---

## 💡 Final Note

This project demonstrates how **data structures + geospatial intelligence + UI/UX** can be combined to build real-world emergency response systems.

---
