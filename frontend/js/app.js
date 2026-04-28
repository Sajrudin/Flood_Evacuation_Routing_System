// ─────────────────────────────────────────
// STATE
// ─────────────────────────────────────────
let src = null, dst = null;
let srcMarker, dstMarker, routeLayer;
let geocoder;

// ─────────────────────────────────────────
// MAP SETUP
// ─────────────────────────────────────────
const map = L.map("map", { zoomControl: true }).setView([30.32, 78.03], 12);

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: "© OpenStreetMap contributors"
}).addTo(map);

map.zoomControl.setPosition("topright");

// Geocoder
geocoder = L.Control.Geocoder.nominatim();

// ─────────────────────────────────────────
// ICONS
// ─────────────────────────────────────────
const srcIcon = L.divIcon({
  html: '<div style="background:#16a34a;width:18px;height:18px;border-radius:50%;border:3px solid #fff;"></div>',
  iconSize:[18,18], iconAnchor:[9,9]
});

const dstIcon = L.divIcon({
  html: '<div style="background:#dc2626;width:18px;height:18px;border-radius:50%;border:3px solid #fff;"></div>',
  iconSize:[18,18], iconAnchor:[9,9]
});

// ─────────────────────────────────────────
// MAP CLICK
// ─────────────────────────────────────────
map.on("click", function(e){
  if (!src) setSource(e.latlng, "Selected location");
  else if (!dst) setDest(e.latlng, "Selected location");
});

function setSource(latlng, label) {
  src = latlng;
  if (srcMarker) map.removeLayer(srcMarker);

  srcMarker = L.marker(latlng, { icon: srcIcon, draggable: true })
    .addTo(map)
    .bindPopup("📍 Source: " + label);

  srcMarker.on("dragend", e => src = e.target.getLatLng());

  document.getElementById("srcInput").value = label;
  document.getElementById("srcClear").classList.remove("hidden");
  closeResults("src");
}

function setDest(latlng, label) {
  dst = latlng;
  if (dstMarker) map.removeLayer(dstMarker);

  dstMarker = L.marker(latlng, { icon: dstIcon, draggable: true })
    .addTo(map)
    .bindPopup("📍 Destination: " + label);

  dstMarker.on("dragend", e => dst = e.target.getLatLng());

  document.getElementById("dstInput").value = label;
  document.getElementById("dstClear").classList.remove("hidden");
  closeResults("dst");
}

// ─────────────────────────────────────────
// AUTOCOMPLETE
// ─────────────────────────────────────────
let debounceTimer;

function onInput(which) {
  const input = document.getElementById(which + "Input");
  const val = input.value.trim();

  if (val.length < 3) { closeResults(which); return; }

  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    geocoder.geocode(val + " Dehradun India", results => {
      showResults(which, results);
    });
  }, 350);
}

function showResults(which, results) {
  const container = document.getElementById(which + "Results");
  container.innerHTML = "";

  if (!results || results.length === 0) return;

  results.slice(0, 5).forEach(r => {
    const item = document.createElement("div");
    item.className = "autocomplete-item";

    const name = r.name || r.html || "";
    item.innerHTML = `<span>${name}</span>`;

    item.onclick = () => {
      const latlng = r.center;
      if (which === "src") setSource(latlng, name);
      else setDest(latlng, name);
    };

    container.appendChild(item);
  });

  container.classList.add("open");
}

function closeResults(which) {
  document.getElementById(which + "Results").classList.remove("open");
}

// ─────────────────────────────────────────
// ROUTE API CALL
// ─────────────────────────────────────────
function findRoute() {
  if (!src || !dst) {
    showStatus("error", "⚠️ Please select source and destination.");
    return;
  }

  setLoading(true);

  const riskThreshold = parseFloat(document.getElementById("riskThresh").value);
  const routePref     = document.getElementById("routePref").value;

  fetch("route", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      source: [src.lat, src.lng],
      destination: [dst.lat, dst.lng],
      risk_threshold: riskThreshold,
      route_preference: routePref
    })
  })
  .then(res => res.json())
  .then(data => {
    setLoading(false);

    if (data.error) {
      showStatus("error", data.error);
      return;
    }

    if (routeLayer) map.removeLayer(routeLayer);

    routeLayer = L.geoJSON(data, {
      style: f => ({
        color: f.properties.risk_score > 0.7 ? "red" : "green",
        weight: 5
      }),
      coordsToLatLng: coords => new L.LatLng(coords[1], coords[0])
    }).addTo(map);

    map.fitBounds(routeLayer.getBounds());

    document.getElementById("sumDist").textContent = data.properties.distance_km + " km";
    document.getElementById("sumTime").textContent = data.properties.travel_time_min + " min";
  })
  .catch(() => {
    setLoading(false);
    showStatus("error", "❌ Is FastAPI server running?");
  });
}

// ─────────────────────────────────────────
// UI HELPERS
// ─────────────────────────────────────────
function setLoading(on) {
  const btn = document.getElementById("routeBtn");

  if (on) {
    btn.classList.add("loading");
    btn.innerHTML = "Loading...";
  } else {
    btn.classList.remove("loading");
    btn.innerHTML = "Find Safe Route";
  }
}

function showStatus(type, msg) {
  const el = document.getElementById("statusMsg");
  el.className = "status-msg " + type;
  el.textContent = msg;
}

// ─────────────────────────────────────────
// HEALTH CHECK
// ─────────────────────────────────────────
function pingHealth() {
  fetch("health")
    .then(() => {
      document.querySelector(".live-dot").style.background = "#22c55e";
    })
    .catch(() => {
      document.querySelector(".live-dot").style.background = "red";
    });
}

setInterval(pingHealth, 10000);