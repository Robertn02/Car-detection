// Project page: builds the demo player, its timeline and map, and the result charts from SITE (written by
// scripts/build_site.py). Demo links: #<demo name> opens that demo.
(() => {
  const $ = s => document.querySelector(s);
  const el = (tag, attrs = {}, ...children) => {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "style") node.style.cssText = v; else if (k in node && k !== "list") node[k] = v; else node.setAttribute(k, v);
    }
    for (const child of children.flat()) node.append(child);
    return node;
  };
  const clock = t => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  const svgNS = "http://www.w3.org/2000/svg";
  const svg = (tag, attrs = {}, text) => {
    const node = document.createElementNS(svgNS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (text !== undefined) node.textContent = text;
    return node;
  };

  // the five relations under the street diagram
  for (const r of SITE.relations)
    $("#keys").append(el("li", {}, el("span", {className: "swatch", style: `background:${r.colour}`}), r.label));

  // ---------------------------------------------------------------------------------------------- demo player
  const video = $("#player"), track = $("#track"), head = $("#head");
  let current = null, map = null, mapLayers = [], dot = null, frame = 0;

  function tiles(target) {
    const m = L.map(target, {scrollWheelZoom: false, attributionControl: true, zoomControl: false});
    L.tileLayer(SITE.tiles.url, {attribution: SITE.tiles.credit, maxZoom: 19}).addTo(m);
    return m;
  }

  function drawMap(demo) {
    const has = !!demo.gps && typeof L !== "undefined";
    $("#nomap").hidden = has;
    $("#minimap").hidden = !has;
    if (!has) return;
    if (!map) map = tiles("minimap");
    mapLayers.forEach(layer => layer.remove());
    const route = demo.gps.route.map(line => L.polyline(line, {color: dark ? "#8c979d" : "#7d878c", weight: 3, opacity: .8}).addTo(map));
    const clipLine = L.polyline(demo.gps.clip.filter(Boolean), {color: getComputedStyle(document.body).getPropertyValue("--green").trim() || "#1f7445", weight: 6}).addTo(map);
    clipLine.on("click", e => {  // jump to the nearest second of the clip
      let best = 0, d = Infinity;
      demo.gps.clip.forEach((p, i) => { if (p) { const dd = (p[0] - e.latlng.lat) ** 2 + (p[1] - e.latlng.lng) ** 2; if (dd < d) { d = dd; best = i; } } });
      seek(best);
    });
    dot = L.circleMarker(demo.gps.clip.find(Boolean), {radius: 8, color: "#fff", weight: 3, fillColor: "#e2b21f", fillOpacity: 1}).addTo(map);
    mapLayers = [...route, clipLine, dot];
    map.invalidateSize();
    map.fitBounds(clipLine.getBounds().pad(0.6));
  }

  function follow() {
    if (!current) return;
    const t = video.currentTime, d = current.duration;
    head.style.left = `${Math.min(100, (t / d) * 100)}%`;
    if (current.gps && dot) {
      const c = current.gps.clip, i = Math.max(0, Math.min(c.length - 1, Math.floor(t))), a = c[i], b = c[i + 1] || a, f = t - i;
      if (a) {
        const p = [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
        dot.setLatLng(p);
        if (!map.getBounds().pad(-0.2).contains(p)) map.panTo(p);
      }
    }
  }
  const loop = () => { follow(); frame = requestAnimationFrame(loop); };
  video.addEventListener("play", () => { cancelAnimationFrame(frame); loop(); });
  video.addEventListener("pause", () => cancelAnimationFrame(frame));
  video.addEventListener("seeked", follow);

  function seek(t) { video.currentTime = Math.max(0, t); video.play().catch(() => {}); }
  track.addEventListener("click", e => {
    if (e.target.closest(".mark") || !current) return;
    const box = track.getBoundingClientRect();
    seek(((e.clientX - box.left) / box.width) * current.duration);
  });

  function select(index, scroll) {
    const demo = SITE.demos[index];
    current = demo;
    video.src = demo.video; video.poster = demo.poster;
    $("#demo-title").textContent = demo.title;
    $("#demo-look").textContent = demo.look;
    $("#duration").textContent = clock(demo.duration);
    $("#demo-meta").replaceChildren(`Ride ${demo.ride}, from ${clock(demo.start)} of the ride. `,
      el("a", {href: demo.video, download: ""}, "Download the clip"));
    track.querySelectorAll(".mark").forEach(m => m.remove());
    const list = $("#moments"); list.replaceChildren();
    for (const m of demo.moments) {
      const kind = SITE.kinds[m.kind];
      const label = `${clock(m.t)} ${m.text}`;
      const mark = el("button", {className: m.span ? "mark span" : "mark", title: label, "aria-label": label,
        style: `left:${(m.t / demo.duration) * 100}%;background:${kind.colour}` + (m.span ? `;width:${(m.span / demo.duration) * 100}%` : "")});
      mark.addEventListener("click", () => seek(m.t - 1));
      track.append(mark);
      const row = el("button", {}, el("time", {}, clock(m.t)), el("span", {className: "dot", style: `background:${kind.colour}`}), m.text);
      row.addEventListener("click", () => seek(m.t - 1));
      list.append(el("li", {}, row));
    }
    if (!demo.moments.length) list.append(el("li", {className: "meta"}, "Nothing flagged in this clip: ordinary riding."));
    document.querySelectorAll(".pick").forEach((p, i) => p.setAttribute("aria-current", i === index ? "true" : "false"));
    head.style.left = "0%";
    drawMap(demo);
    if (scroll) $("#demos").scrollIntoView();
  }

  SITE.demos.forEach((demo, i) => {
    const pick = el("button", {className: "pick", role: "listitem", "aria-current": "false"},
      el("img", {src: demo.poster, alt: "", loading: "lazy", width: 320, height: 180}),
      el("span", {}, demo.title),
      el("div", {className: "tags"}, demo.tags.map(t => el("small", {className: "tag"}, t))));
    pick.addEventListener("click", () => { history.replaceState(null, "", `#${demo.name}`); select(i, false); video.play().catch(() => {}); });
    $("#playlist").append(pick);
  });
  const fromHash = SITE.demos.findIndex(d => `#${d.name}` === location.hash);
  const startDemo = () => select(Math.max(0, fromHash), fromHash >= 0);
  if (typeof L !== "undefined") startDemo(); else window.addEventListener("load", startDemo);

  // ---------------------------------------------------------------------------------------------- results
  $("#results-lede").textContent = SITE.resultsLede;
  $("#fineprint").textContent = SITE.fineprint;
  for (const f of SITE.figures)
    $("#figures").append(el("div", {}, el("dt", {className: "figure-value"}, f.value), el("dd", {className: "figure-label"}, f.label)));

  // vehicles per minute: a row per relation, a column per riding context, each bar scaled within its row
  (() => {
    const table = $("#exposure"), cols = SITE.exposure;
    const head = el("tr", {}, el("th", {scope: "col"}, el("span", {className: "visually-hidden"}, "Relation")),
      cols.map(c => el("th", {scope: "col", style: `border-top-color:${c.colour}`}, c.label,
        el("small", {}, `${Math.round(c.minutes)} min of riding`))));
    table.append(el("thead", {}, head));
    const body = el("tbody");
    for (const r of SITE.relations) {
      const values = cols.map(c => c.per_min[r.key] || 0), top = Math.max(...values, 1e-6);
      body.append(el("tr", {}, el("th", {scope: "row"}, el("span", {className: "swatch", style: `background:${r.colour}`}), r.label),
        values.map(v => el("td", {}, el("span", {className: "num"}, v.toFixed(1)),
          el("span", {className: "meter", "aria-hidden": "true"}, el("i", {style: `width:${(v / top) * 100}%;background:${r.colour}`}))))));
    }
    table.append(body);
  })();

  // all routes, on a small map, once the panel is in view
  const routes = $("#routes");
  new IntersectionObserver((entries, observer) => {
    if (!entries.some(e => e.isIntersecting) || typeof L === "undefined") return;
    observer.disconnect();
    fetch("map/exposure.json").then(r => r.json()).then(data => {
      const m = tiles(routes);
      const lines = data.routes.flatMap(r => r.lines.map(line => L.polyline(line, {color: "#1f7445", weight: 3, opacity: .85}).addTo(m)));
      if (lines.length) m.fitBounds(L.featureGroup(lines).getBounds().pad(0.05));
    }).catch(() => routes.remove());
  }, {rootMargin: "200px"}).observe(routes);
})();
