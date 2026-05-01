// STATE

let src = null,
  dst = null;
let srcMarker, dstMarker;
let routeLayers = {};
let geocoder;

// MAP SETUP

const map = L.map("map").setView([30.32, 78.03], 12);

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: "© OpenStreetMap contributors",
}).addTo(map);

geocoder = L.Control.Geocoder.nominatim();

// ICONS

const srcIcon = L.divIcon({
  html: '<div style="background:#16a34a;width:18px;height:18px;border-radius:50%;border:3px solid white;"></div>',
  iconSize: [18, 18],
  iconAnchor: [9, 9],
});

const dstIcon = L.divIcon({
  html: '<div style="background:#dc2626;width:18px;height:18px;border-radius:50%;border:3px solid white;"></div>',
  iconSize: [18, 18],
  iconAnchor: [9, 9],
});


// MAP CLICK

map.on("click", (e) => {
  if (!src) setSource(e.latlng, "Selected location");
  else if (!dst) setDest(e.latlng, "Selected location");
});

// ─────────────────────────────────────────
// SET SOURCE / DEST
// ─────────────────────────────────────────
function setSource(latlng, label) {
  src = latlng;

  if (srcMarker) map.removeLayer(srcMarker);

  srcMarker = L.marker(latlng, { icon: srcIcon, draggable: true })
    .addTo(map)
    .bindPopup("Source");

  srcMarker.on("dragend", (e) => (src = e.target.getLatLng()));

  document.getElementById("srcInput").value = label;
}

function setDest(latlng, label) {
  dst = latlng;

  if (dstMarker) map.removeLayer(dstMarker);

  dstMarker = L.marker(latlng, { icon: dstIcon, draggable: true })
    .addTo(map)
    .bindPopup("Destination");

  dstMarker.on("dragend", (e) => (dst = e.target.getLatLng()));

  document.getElementById("dstInput").value = label;
}

// AUTOCOMPLETE

let debounceTimer;

function onInput(which) {
  const val = document.getElementById(which + "Input").value;

  if (val.length < 3) return;

  clearTimeout(debounceTimer);

  debounceTimer = setTimeout(() => {
    geocoder.geocode(val + " Dehradun India", (results) => {
      const container = document.getElementById(which + "Results");
      container.innerHTML = "";

      results.slice(0, 5).forEach((r) => {
        const item = document.createElement("div");
        item.innerHTML = r.name || r.html;

        item.onclick = () => {
          if (which === "src") setSource(r.center, r.name);
          else setDest(r.center, r.name);
        };

        container.appendChild(item);
      });
    });
  }, 300);
}

// ─────────────────────────────────────────
// FETCH ROUTES
// ─────────────────────────────────────────
async function findRoute() {
  if (!src || !dst) {
    showStatus("error", "Select source and destination");
    return;
  }

  setLoading(true);

  try {
    const res = await fetch("/route", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        source: [src.lat, src.lng],
        destination: [dst.lat, dst.lng],
      }),
    });

    const data = await res.json();

    if (data.error) {
      showStatus("error", data.error);
      return;
    }

    drawRoutes(data);

    showStatus("success", "Routes loaded");
  } catch (err) {
    console.error(err);
    showStatus("error", "Server error");
  }

  setLoading(false);
}

// ─────────────────────────────────────────
// DRAW ROUTES
// ─────────────────────────────────────────
function drawRoutes(routes) {
  // Clear old
  Object.values(routeLayers).forEach((layer) => map.removeLayer(layer));
  routeLayers = {};

  const colors = {
    shortest: "blue",
    safe: "green",
    balanced: "orange",
    unsafe: "red",
  };

  for (let key in routes) {
    if (
      !routes[key] ||
      !routes[key].features ||
      routes[key].features.length === 0
    ) {
      console.warn(key + " route is empty");
      continue;
    }

    const layer = L.geoJSON(routes[key], {
      style: {
        color: colors[key],
        weight: key === "safe" ? 6 : 4,
      },
    }).addTo(map);

    routeLayers[key] = layer;
  }

  const mainRoute = routeLayers.safe || routeLayers.shortest;

  if (mainRoute) {
    const bounds = mainRoute.getBounds();
    if (bounds.isValid()) {
      map.fitBounds(bounds);
    }
  }
}

// ─────────────────────────────────────────
// TOGGLE ROUTE
// ─────────────────────────────────────────
function toggleRoute(type) {
  if (!routeLayers[type]) return;

  if (map.hasLayer(routeLayers[type])) {
    map.removeLayer(routeLayers[type]);
  } else {
    routeLayers[type].addTo(map);
  }
}

// ─────────────────────────────────────────
// RESET MAP
// ─────────────────────────────────────────
function resetMap() {
  src = null;
  dst = null;

  if (srcMarker) map.removeLayer(srcMarker);
  if (dstMarker) map.removeLayer(dstMarker);

  Object.values(routeLayers).forEach((layer) => map.removeLayer(layer));
  routeLayers = {};

  document.getElementById("srcInput").value = "";
  document.getElementById("dstInput").value = "";
}

// ─────────────────────────────────────────
// UI HELPERS
// ─────────────────────────────────────────
function setLoading(on) {
  const btn = document.getElementById("routeBtn");

  btn.innerText = on ? "Loading..." : "Find Routes";
  btn.disabled = on;
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
  fetch("/health")
    .then(
      () => (document.querySelector(".live-dot").style.background = "green"),
    )
    .catch(
      () => (document.querySelector(".live-dot").style.background = "red"),
    );
}

setInterval(pingHealth, 10000);
