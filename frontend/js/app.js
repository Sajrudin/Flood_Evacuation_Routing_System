// ─────────────────────────────────────────
// STATE
// ─────────────────────────────────────────
let src = null, dst = null;
let srcMarker, dstMarker;
let routeLayers = {};
let geocoder;

let currentRoutes = {};
let activeRouteType = "safe";

// ─────────────────────────────────────────
// MAP SETUP
// ─────────────────────────────────────────
const map = L.map("map").setView([30.32, 78.03], 12);

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: "© OpenStreetMap contributors",
}).addTo(map);

geocoder = L.Control.Geocoder.nominatim();

// ─────────────────────────────────────────
// ICONS
// ─────────────────────────────────────────
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

// ─────────────────────────────────────────
// MAP CLICK
// ─────────────────────────────────────────
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
  document.getElementById("srcResults").innerHTML = "";
  document.getElementById("srcClear").classList.remove("hidden");
}

function setDest(latlng, label) {
  dst = latlng;

  if (dstMarker) map.removeLayer(dstMarker);

  dstMarker = L.marker(latlng, { icon: dstIcon, draggable: true })
    .addTo(map)
    .bindPopup("Destination");

  dstMarker.on("dragend", (e) => (dst = e.target.getLatLng()));

  document.getElementById("dstInput").value = label;
  document.getElementById("dstResults").innerHTML = "";
  document.getElementById("dstClear").classList.remove("hidden");
}

function clearPoint(which) {
  if (which === "src") {
    src = null;
    if (srcMarker) map.removeLayer(srcMarker);
    document.getElementById("srcInput").value = "";
    document.getElementById("srcResults").innerHTML = "";
    document.getElementById("srcClear").classList.add("hidden");
  } else {
    dst = null;
    if (dstMarker) map.removeLayer(dstMarker);
    document.getElementById("dstInput").value = "";
    document.getElementById("dstResults").innerHTML = "";
    document.getElementById("dstClear").classList.add("hidden");
  }
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
  showStatus("", "");

  try {
    const res = await fetch("/route", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
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
    showStatus("success", "Routes loaded successfully");
  } catch (err) {
    console.error(err);
    showStatus("error", "Server error");
  } finally {
    setLoading(false);
  }
}

// ─────────────────────────────────────────
// DRAW ROUTES (UPDATED WITH HOVER)
// ─────────────────────────────────────────
function drawRoutes(routes) {
  currentRoutes = routes;

  Object.values(routeLayers).forEach((layer) => map.removeLayer(layer));
  routeLayers = {};

  const colors = {
    shortest: "#3b82f6",
    safe: "#10b981",
    balanced: "#f59e0b",
    unsafe: "#ef4444",
  };

  for (let key in routes) {
    if (!routes[key] || !routes[key].features) continue;

    const layer = L.geoJSON(routes[key], {
      style: {
        color: colors[key],
        weight: key === "safe" ? 7 : 5,
        opacity: 0.8,
      },

      onEachFeature: function (feature, l) {
        const p = feature.properties || {};
        const length = ((p.length || 0) / 1000).toFixed(2);
        const risk = (p.risk_score || 0).toFixed(3);
        const blocked = p.blocked ? "Yes" : "No";

        const tooltip = `
          <b>${key.toUpperCase()}</b><br>
          Segment: ${length} km<br>
          Risk: ${risk}<br>
          Blocked: ${blocked}
        `;

        l.bindTooltip(tooltip, {
          sticky: true,
          className: "route-tooltip",
        });

        // Highlight on hover
        l.on("mouseover", function () {
          this.setStyle({ weight: 8, opacity: 1 });
        });

        l.on("mouseout", function () {
          this.setStyle({
            weight: key === "safe" ? 7 : 5,
            opacity: 0.8,
          });
        });
      },
    });

    // Popup with full route summary
    const summary = getRouteSummary(routes[key]);
    layer.bindPopup(`
      <b>${key.toUpperCase()} ROUTE</b><br>
      Distance: ${summary.distance} km<br>
      Time: ${summary.time} min<br>
      Avg Risk: ${summary.risk}
    `);

    routeLayers[key] = layer;
  }

  // Show default route
  if (routeLayers.safe) {
    routeLayers.safe.addTo(map);
    activeRouteType = "safe";
  }

  const bounds = routeLayers[activeRouteType].getBounds();
  map.fitBounds(bounds, { padding: [50, 50] });

  updateSummaryForType(activeRouteType);
}

// ─────────────────────────────────────────
// TOGGLE ROUTE
// ─────────────────────────────────────────
function toggleRoute(type) {
  if (!routeLayers[type]) return;

  Object.keys(routeLayers).forEach((key) => {
    if (map.hasLayer(routeLayers[key])) {
      map.removeLayer(routeLayers[key]);
    }
  });

  routeLayers[type].addTo(map);
  activeRouteType = type;
  updateSummaryForType(type);
}

// ─────────────────────────────────────────
// SUMMARY
// ─────────────────────────────────────────
function updateSummaryForType(type) {
  const route = currentRoutes[type];
  if (!route || !route.features) return;

  let total = 0;
  let unsafe = 0;

  route.features.forEach(f => {
    const p = f.properties || {};
    total += p.length || 0;
    if (p.blocked || p.risk_score > 0.6) unsafe++;
  });

  const km = (total / 1000).toFixed(2);
  const time = Math.round((total / 1000) / 25 * 60);

  document.getElementById("sumDist").textContent = `${km} km`;
  document.getElementById("sumTime").textContent = `~${time} min`;
  document.getElementById("sumUnsafe").textContent = unsafe;
}

// ─────────────────────────────────────────
// ROUTE SUMMARY HELPER
// ─────────────────────────────────────────
function getRouteSummary(route) {
  let total = 0;
  let riskSum = 0;
  let count = 0;

  route.features.forEach(f => {
    const p = f.properties || {};
    total += p.length || 0;
    riskSum += p.risk_score || 0;
    count++;
  });

  return {
    distance: (total / 1000).toFixed(2),
    time: Math.round((total / 1000) / 25 * 60),
    risk: (riskSum / count).toFixed(3)
  };
}

// ─────────────────────────────────────────
// RESET
// ─────────────────────────────────────────
function resetMap() {
  src = null;
  dst = null;

  if (srcMarker) map.removeLayer(srcMarker);
  if (dstMarker) map.removeLayer(dstMarker);

  Object.values(routeLayers).forEach((layer) => map.removeLayer(layer));
  routeLayers = {};
  currentRoutes = {};

  document.getElementById("sumDist").textContent = "—";
  document.getElementById("sumTime").textContent = "—";
  document.getElementById("sumUnsafe").textContent = "—";
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
    .then((r) => {
      document.querySelector(".live-dot").style.background = r.ok ? "green" : "red";
    })
    .catch(() => {
      document.querySelector(".live-dot").style.background = "red";
    });
}

pingHealth();
setInterval(pingHealth, 10000);