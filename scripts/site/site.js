// Project page: the clip player with its timeline and map, the tables and the figures, from SITE (written by
// scripts/build_site.py). Links: #<demo name> opens that clip, #<demo name>@<seconds> at that moment.
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
  const percent = x => `${Math.round(x * 100)}%`;
  const ink = (node = document.body) => getComputedStyle(node).getPropertyValue("--ink").trim() || "#16191b";

  // Figure 2: what each relation means, under the street
  for (const r of SITE.relations)
    $("#keys").append(el("div", {style: `--c:${r.colour}`}, el("dt", {}, r.label), el("dd", {}, r.meaning)));

  // ---------------------------------------------------------------------------------------------- Figure 1: the clips
  const video = $("#player"), track = $("#track"), head = $("#head"), list = $("#moments");
  let current = null, map = null, mapLayers = [], dot = null, frame = 0;

  // the map's tiles; where tile.openstreetmap.org cannot be reached (some networks do not resolve it, and take seconds
  // to say so) - no tile in TILE_WAIT_MS, or more failing than loading (a few may come from the browser's cache) - the
  // same map data from a second server
  const TILE_WAIT_MS = 3000;
  function baseTiles(m, tiles) {
    const layer = L.tileLayer(tiles.url, {attribution: tiles.credit, maxZoom: 19}).addTo(m);
    let loads = 0, errors = 0, swapped = false;
    const swap = () => {
      if (swapped || !tiles.fallback) return;
      swapped = true;
      m.attributionControl.removeAttribution(tiles.credit).addAttribution(tiles.fallback.credit);
      layer.options.attribution = tiles.fallback.credit;
      layer.setUrl(tiles.fallback.url);
    };
    layer.on("tileload", () => { loads++; });
    layer.on("tileerror", () => { if (++errors > loads) swap(); });
    setTimeout(() => { if (!loads) swap(); }, TILE_WAIT_MS);
    return layer;
  }

  function tiles(target) {
    const m = L.map(target, {scrollWheelZoom: false, attributionControl: true, zoomControl: false});
    m.attributionControl.setPrefix(false);
    baseTiles(m, SITE.tiles);
    return m;
  }

  function drawMap(demo) {
    const has = !!demo.gps && typeof L !== "undefined";
    $(".side").classList.toggle("nomap", !has);  // a ride without GPS: the list takes the map's room
    if (!has) return;
    if (!map) map = tiles("minimap");
    mapLayers.forEach(layer => layer.remove());
    const route = demo.gps.route.map(line => L.polyline(line, {color: "#7f8a90", weight: 2, opacity: .9}).addTo(map));
    const clipLine = L.polyline(demo.gps.clip.filter(Boolean), {color: ink($(".band")), weight: 5}).addTo(map);
    clipLine.on("click", e => {  // jump to the nearest second of the clip
      let best = 0, d = Infinity;
      demo.gps.clip.forEach((p, i) => { if (p) { const dd = (p[0] - e.latlng.lat) ** 2 + (p[1] - e.latlng.lng) ** 2; if (dd < d) { d = dd; best = i; } } });
      seek(best);
    });
    dot = L.circleMarker(demo.gps.clip.find(Boolean), {radius: 7, color: "#fff", weight: 3, fillColor: "#e2b21f", fillOpacity: 1}).addTo(map);
    mapLayers = [...route, clipLine, dot];
    map.invalidateSize();
    map.fitBounds(clipLine.getBounds().pad(0.6));
  }

  function follow() {
    if (!current) return;
    const t = video.currentTime, d = current.duration;
    head.style.left = `${Math.min(100, (t / d) * 100)}%`;
    // the moment the clip is at: the last one that has begun
    let now = -1;
    current.moments.forEach((m, i) => { if (m.t - 1 <= t) now = i; });
    [...list.children].forEach((li, i) => li.setAttribute("aria-current", i === now ? "true" : "false"));
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

  function seek(t) {  // before the clip's length is known, a seek is lost: wait for it
    const go = () => { video.currentTime = Math.max(0, Math.min(t, video.duration - 0.1)); video.play().catch(() => {}); };
    if (video.readyState >= 1) go(); else video.addEventListener("loadedmetadata", go, {once: true});
  }
  track.addEventListener("click", e => {
    if (e.target.closest(".mark") || !current) return;
    const box = track.getBoundingClientRect();
    seek(((e.clientX - box.left) / box.width) * current.duration);
  });

  function select(index) {
    const demo = SITE.demos[index];
    current = demo;
    video.src = demo.video; video.poster = demo.poster;
    $("#demo-title").textContent = demo.title;
    $("#demo-look").textContent = demo.look;
    $("#duration").textContent = clock(demo.duration);
    $("#demo-meta").replaceChildren(`Ride ${demo.ride}, from ${clock(demo.start)} to ${clock(demo.start + demo.duration)}` +
      `${demo.gps ? "" : ", no GPS track"}. `, el("a", {href: demo.video, download: ""}, "Download the clip"));
    track.querySelectorAll(".mark").forEach(m => m.remove());
    list.replaceChildren();
    for (const m of demo.moments) {
      const kind = SITE.kinds[m.kind];
      const label = `${clock(m.t)} ${m.text}${m.why ? ". " + m.why : ""}`;
      const mark = el("button", {className: "mark" + (m.span ? " span" : "") + (m.why ? " said" : ""), title: label, "aria-label": label,
        style: `left:${(m.t / demo.duration) * 100}%;background:${kind.colour}` + (m.span ? `;width:${(m.span / demo.duration) * 100}%` : "")});
      const lead = m.why ? 8 : 1;  // a pass whose cause a rider gave plays from before the cause
      mark.addEventListener("click", () => seek(m.t - lead));
      track.append(mark);
      const row = el("button", {title: kind.label},
        el("time", {}, clock(m.t)), el("span", {className: "dot", style: `background:${kind.colour}`}), el("span", {}, m.text),
        m.why ? el("span", {className: "why"}, m.why) : []);
      row.addEventListener("click", () => seek(m.t - lead));
      list.append(el("li", {"aria-current": "false"}, row));
    }
    if (!demo.moments.length) list.append(el("li", {className: "none"}, "Nothing flagged in this clip."));
    document.querySelectorAll(".pick").forEach((p, i) => p.setAttribute("aria-current", i === index ? "true" : "false"));
    head.style.left = "0%";
    drawMap(demo);
  }

  SITE.demos.forEach((demo, i) => {
    const pick = el("button", {className: "pick", role: "listitem", "aria-current": "false"},
      el("img", {src: demo.poster, alt: "", loading: "lazy", width: 320, height: 180}),
      el("strong", {}, demo.title), demo.summary);
    pick.addEventListener("click", () => { history.replaceState(null, "", `#${demo.name}`); select(i); video.play().catch(() => {}); });
    $("#playlist").append(pick);
  });
  function open(hash, scroll) {  // #<demo name> or #<demo name>@<seconds>
    const [name, at] = hash.slice(1).split("@");
    const index = SITE.demos.findIndex(d => d.name === name);
    if (index < 0) return false;
    select(index);
    if (at !== undefined) seek(Number(at));
    if (scroll) $("#demos").scrollIntoView();
    return true;
  }
  const start = () => { if (!open(location.hash, true)) select(0); };
  if (typeof L !== "undefined") start(); else window.addEventListener("load", start);
  window.addEventListener("hashchange", () => open(location.hash, true));

  // ---------------------------------------------------------------------------------------------- tables
  const cell = (text, numeric) => el("td", numeric ? {className: "n"} : {}, text);
  (() => {  // Table 1: the rides
    const head = [["Ride", 0], ["When", 0], ["Minutes", 1], ["Vehicles", 1], ["Overtakes", 1], ["Under 1.5 m", 1], ["Red lights", 1], ["Door zone", 1]];
    $("#rides").append(el("thead", {}, el("tr", {}, head.map(([h, n]) => el("th", {scope: "col", className: n ? "n" : ""}, h)))));
    const body = el("tbody");
    SITE.rides.forEach((r, i) => body.append(el("tr", {className: i === SITE.rides.length - 1 ? "total" : ""},
      el("th", {scope: "row"}, r.ride), el("td", {className: "muted"}, r.when), cell(r.minutes, 1), cell(r.vehicles.toLocaleString("en-US"), 1),
      cell(r.overtakes, 1), cell(r.close, 1), cell(r.red_waits, 1), cell(percent(r.door), 1))));
    $("#rides").append(body);
  })();

  (() => {  // Table 2: vehicles a minute, a row per relation, a column per riding context, bars scaled within the row
    const table = $("#exposure"), cols = SITE.exposure;
    table.append(el("thead", {}, el("tr", {}, el("th", {scope: "col"}, "Relation"),
      cols.map(c => el("th", {scope: "col"}, c.label, el("small", {}, `${Math.round(c.minutes)} min of riding`))))));
    const body = el("tbody");
    for (const r of SITE.relations) {
      const values = cols.map(c => c.per_min[r.key] || 0), top = Math.max(...values, 1e-6);
      body.append(el("tr", {}, el("th", {scope: "row"}, el("span", {className: "swatch", style: `background:${r.colour}`}), r.label),
        values.map(v => el("td", {}, el("span", {className: "num"}, v.toFixed(1)),
          el("span", {className: "meter", "aria-hidden": "true"}, el("i", {style: `width:${(v / top) * 100}%;background:${r.colour}`}))))));
    }
    table.append(body);
  })();

  (() => {  // Table 3: how often the label is right, by how close the vehicle came
    const a = SITE.accuracy, table = $("#accuracy");
    if (!a.by_distance) { table.closest("figure").remove(); return; }
    table.append(el("thead", {}, el("tr", {}, el("th", {scope: "col"}, "Closest approach"),
      el("th", {scope: "col", className: "n"}, "Vehicles"), el("th", {scope: "col", className: "n"}, "Right"))));
    const body = el("tbody");
    for (const d of a.by_distance) body.append(el("tr", {}, el("th", {scope: "row"}, d.range), cell(d.n, 1), cell(percent(d.right), 1)));
    body.append(el("tr", {className: "total"}, el("th", {scope: "row"}, `Within ${Math.round(a.limit_m)} m`), cell(a.n, 1), cell(percent(a.right), 1)));
    table.append(body);
  })();

  // ---------------------------------------------------------------------------------------------- Figure 3: the case
  (() => {
    const c = SITE.context;
    for (const s of c.stills)
      $("#stills").append(el("figure", {}, el("img", {src: s.src, alt: s.text, loading: "lazy", width: 1280, height: 720}),
        el("figcaption", {}, el("b", {}, s.time), " ", s.text)));
    $("#case-play").addEventListener("click", e => {
      e.preventDefault();
      history.replaceState(null, "", `#${c.demo}@${c.at}`);
      open(location.hash, true);
    });
  })();

  // ---------------------------------------------------------------------------------------------- Figure 4: the routes
  const routes = $("#routes");
  new IntersectionObserver((entries, observer) => {
    if (!entries.some(e => e.isIntersecting) || typeof L === "undefined") return;
    observer.disconnect();
    fetch("map/exposure.json").then(r => r.json()).then(data => {
      const m = tiles(routes);
      const lines = data.routes.flatMap(r => r.lines.map(line => L.polyline(line, {color: ink(), weight: 3, opacity: .9}).addTo(m)));
      if (lines.length) m.fitBounds(L.featureGroup(lines).getBounds().pad(0.08));
    }).catch(() => routes.closest("figure").remove());
  }, {rootMargin: "200px"}).observe(routes);
})();
