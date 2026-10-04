// GW1B 3D model visualizer — architecture as stacked slabs, cells coloured by the weights of a checkpoint.
// Data comes from the server API (python -m gw1b.viz.server) or, for a self-contained page, from <script id="gw1b-data">.
import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const state = {
  data: null, runs: [], run: null, step: null,
  mode: 'rms', scale: 'tensor', spacing: 1, labels: true, follow: true,
  slabs: [], hovered: null, timer: null, embedded: false,
};

// ---------------------------------------------------------------------------------------------
// colour maps (0..1 -> rgb)
// ---------------------------------------------------------------------------------------------
const STOPS = {
  seq:  [[13, 8, 135], [70, 3, 159], [114, 1, 168], [156, 23, 158], [189, 55, 134], [216, 87, 107], [237, 121, 83], [251, 159, 58], [253, 202, 38], [240, 249, 33]],  // plasma
  heat: [[8, 12, 30], [40, 20, 90], [110, 30, 120], [180, 50, 100], [230, 100, 60], [250, 170, 40], [255, 235, 120]],
  div:  [[33, 102, 172], [103, 169, 207], [209, 229, 240], [247, 247, 247], [253, 219, 199], [239, 138, 98], [178, 24, 43]],  // blue-white-red
};
function cmap(name, t) {
  const s = STOPS[name]; t = Math.min(1, Math.max(0, t)); const x = t * (s.length - 1); const i = Math.min(s.length - 2, Math.floor(x)); const f = x - i;
  return [0, 1, 2].map(k => Math.round(s[i][k] + (s[i + 1][k] - s[i][k]) * f));
}
function modeInfo() {
  return { rms: { name: 'seq', label: 'block RMS of the weights' }, mean: { name: 'div', label: 'block mean of the weights (sign)' },
           delta: { name: 'heat', label: 'block RMS of Δ since prev. checkpoint' } }[state.mode];
}
// normalisation range for a tensor in the current mode
function rangeFor(t, globalHi) {
  if (state.mode === 'mean') { const hi = state.scale === 'global' && t.shape.length > 1 ? globalHi : Math.max(1e-12, ...t.mean.map(Math.abs)); return [-hi, hi]; }
  const vals = state.mode === 'delta' ? t.delta : t.rms;
  if (!vals) return [0, 1];
  if (state.scale === 'global' && t.shape.length > 1) return [0, globalHi];
  const lo = Math.min(...vals), hi = Math.max(...vals);          // per tensor: stretch min..max to show the structure
  return hi - lo > 1e-9 * Math.max(1e-12, hi) ? [lo, hi] : [0, Math.max(1e-12, hi)];
}
function globalHi(data) {
  const mats = data.tensors.filter(t => t.shape.length > 1);
  if (state.mode === 'mean') return Math.max(1e-12, ...mats.map(t => Math.max(...t.mean.map(Math.abs))));
  const key = state.mode === 'delta' ? 'delta' : 'rms';
  return Math.max(1e-12, ...mats.filter(t => t[key]).map(t => Math.max(...t[key])));
}
function valuesFor(t) { return state.mode === 'mean' ? t.mean : state.mode === 'delta' ? t.delta : t.rms; }

function paint(t, lo, hi) {   // -> Uint8Array RGBA, rows x cols (row 0 first)
  const vals = valuesFor(t); const n = t.rows * t.cols; const out = new Uint8Array(n * 4); const cm = modeInfo().name;
  for (let i = 0; i < n; i++) {
    let rgb;
    if (!vals) rgb = [60, 66, 80];
    else rgb = cmap(cm, (vals[i] - lo) / (hi - lo || 1));
    out[i * 4] = rgb[0]; out[i * 4 + 1] = rgb[1]; out[i * 4 + 2] = rgb[2]; out[i * 4 + 3] = 255;
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// three.js scene
// ---------------------------------------------------------------------------------------------
const canvas = $('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0e1117);
const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 5000);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.08; controls.screenSpacePanning = true;
const world = new THREE.Group(); scene.add(world);
const raycaster = new THREE.Raycaster(); const pointer = new THREE.Vector2(-2, -2);
let labelSprites = [];

function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize); resize();

const unit = (n) => 0.75 * Math.log2(Math.max(2, n));   // matrix dimension -> scene units
const sideMat = new THREE.MeshBasicMaterial({ color: 0x1b2230 });
const edgeMat = new THREE.LineBasicMaterial({ color: 0x3a4557, transparent: true, opacity: 0.9 });
const hoverEdgeMat = new THREE.LineBasicMaterial({ color: 0xffffff });

function textSprite(text, size = 1.0, color = '#c9d1e0') {
  const c = document.createElement('canvas'); const ctx = c.getContext('2d');
  const font = '28px system-ui, sans-serif'; ctx.font = font; const w = Math.ceil(ctx.measureText(text).width) + 16; c.width = w; c.height = 40;
  ctx.font = font; ctx.fillStyle = color; ctx.textBaseline = 'middle'; ctx.fillText(text, 8, 20);
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }));
  sp.scale.set(size * w / 40, size, 1); sp.renderOrder = 10; return sp;
}

function slabMesh(t, w, d, lo, hi) {
  const h = 0.14;
  const geom = new THREE.BoxGeometry(w, h, d);
  const tex = new THREE.DataTexture(paint(t, lo, hi), t.cols, t.rows, THREE.RGBAFormat);
  tex.magFilter = THREE.NearestFilter; tex.minFilter = THREE.NearestFilter; tex.colorSpace = THREE.SRGBColorSpace;
  tex.flipY = true; tex.needsUpdate = true;
  const top = new THREE.MeshBasicMaterial({ map: tex });
  const mesh = new THREE.Mesh(geom, [sideMat, sideMat, top, sideMat, sideMat, sideMat]);
  const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geom), edgeMat);
  mesh.add(edges);
  mesh.userData = { tensor: t, tex, top, edges, lo, hi, w, d };
  return mesh;
}

function clearScene() {
  for (const s of state.slabs) { s.userData.tex.dispose(); s.userData.top.dispose(); s.geometry.dispose(); }
  while (world.children.length) world.remove(world.children[0]);
  state.slabs = []; labelSprites = []; state.hovered = null; $('info').style.display = 'none';
}

// layout: embedding at the bottom, one block per layer (QKV / O / gate+up / down), final norm + head on top
function build() {
  clearScene();
  const data = state.data; if (!data) return;
  const gHi = globalHi(data);
  const byLayer = new Map(); const top = [];
  for (const t of data.tensors) {
    if (t.layer === null || t.layer === undefined) top.push(t);
    else { if (!byLayer.has(t.layer)) byLayer.set(t.layer, []); byLayer.get(t.layer).push(t); }
  }
  const lh = 1.5 * state.spacing, gap = 2.4 * state.spacing;
  let y = 0; let maxHalf = 0;
  const place = (t, x, yy, z, w, d) => {
    const [lo, hi] = rangeFor(t, gHi); const m = slabMesh(t, w, d, lo, hi); m.position.set(x, yy, z); world.add(m); state.slabs.push(m);
    maxHalf = Math.max(maxHalf, Math.abs(x) + w / 2);
    if (state.labels) { const sp = textSprite(t.label, 0.55, '#9aa6bb'); sp.position.set(x, yy + 0.35, z + d / 2 + 0.25); world.add(sp); labelSprites.push(sp); }
    return m;
  };
  const level = (tensors, yy) => {   // side by side along x, centred; vectors as thin strips on the left
    const mats = tensors.filter(t => t.shape.length > 1), vecs = tensors.filter(t => t.shape.length === 1);
    const widths = mats.map(t => unit(t.shape[1])); const total = widths.reduce((a, b) => a + b, 0) + 0.5 * (mats.length - 1);
    let x = -total / 2;
    mats.forEach((t, i) => { place(t, x + widths[i] / 2, yy, 0, widths[i], unit(t.shape[0])); x += widths[i] + 0.5; });
    vecs.forEach((t, i) => place(t, -total / 2 - 0.6 - unit(t.shape[0]) / 2, yy, 0, unit(t.shape[0]), 0.35 + i * 0.5));
  };
  const blockLabels = [];
  const blockLabel = (text, yy, big = false) => {   // placed to the right of the widest level once everything is laid out
    const sp = textSprite(text, big ? 1.3 : 0.9, big ? '#e6e9ef' : '#c9d1e0'); sp.position.set(0, yy, 0); world.add(sp); labelSprites.push(sp); blockLabels.push([sp, big]);
  };
  // bottom: embeddings
  const emb = top.filter(t => t.group === 'embed');
  for (const t of emb) { place(t, 0, y, 0, unit(t.shape[1]), unit(t.shape[0])); y += lh; }
  blockLabel('embedding', y - lh + 0.6, true);
  y += gap;
  // blocks
  const layers = [...byLayer.keys()].sort((a, b) => a - b);
  for (const L of layers) {
    const ts = byLayer.get(L); const g = (names) => ts.filter(t => names.includes(t.label));
    const y0 = y;
    level([...g(['norm (attn)']), ...g(['Q']), ...g(['K']), ...g(['V'])], y); y += lh;
    level(g(['O']), y); y += lh;
    level([...g(['norm (mlp)']), ...g(['gate']), ...g(['up'])], y); y += lh;
    level(g(['down']), y); y += lh;
    const rest = ts.filter(t => !['norm (attn)', 'Q', 'K', 'V', 'O', 'norm (mlp)', 'gate', 'up', 'down'].includes(t.label));
    if (rest.length) { level(rest, y); y += lh; }
    blockLabel(`block ${L}`, y0 + 1.5 * lh);
    y += gap;
  }
  // top: final norm, head
  const fin = top.filter(t => t.group === 'norm'); const head = top.filter(t => t.group === 'head');
  if (fin.length) { level(fin, y); y += lh; }
  if (head.length) { for (const t of head) { place(t, 0, y, 0, unit(t.shape[1]), unit(t.shape[0])); y += lh; } blockLabel('output head', y - lh, true); }
  else if (data.model.tie_embeddings && emb.length) {
    const t = emb[0]; const m = place(t, 0, y, 0, unit(t.shape[1]), unit(t.shape[0])); m.userData.top.transparent = true; m.userData.top.opacity = 0.75;
    blockLabel('output (tied to the embedding)', y, true); y += lh;
  }
  for (const [sp, big] of blockLabels) sp.position.x = maxHalf + 0.8 + sp.scale.x / 2;
  state.height = y; state.halfWidth = maxHalf;
  relabel();
  updateLegend(gHi);
}

function relabel() { for (const s of labelSprites) s.visible = state.labels; }

function recolor() {   // mode / scale changed: repaint textures only
  const gHi = globalHi(state.data);
  for (const m of state.slabs) {
    const t = m.userData.tensor; const [lo, hi] = rangeFor(t, gHi);
    m.userData.tex.image.data.set(paint(t, lo, hi)); m.userData.tex.needsUpdate = true; m.userData.lo = lo; m.userData.hi = hi;
  }
  updateLegend(gHi);
  if (state.hovered) showInfo(state.hovered.userData.tensor, state.hovered);
}

function fitCamera() {
  const H = state.height || 10, W = 2 * (state.halfWidth || 5) + 8;
  const fov = THREE.MathUtils.degToRad(camera.fov / 2);
  const dist = Math.max(H / (2 * Math.tan(fov)), W / (2 * Math.tan(fov) * camera.aspect)) * 1.15;
  const az = THREE.MathUtils.degToRad(38), el = THREE.MathUtils.degToRad(24);   // front-right, slightly above
  controls.target.set(0, H / 2, 0);
  camera.position.set(dist * Math.cos(el) * Math.sin(az), H / 2 + dist * Math.sin(el), dist * Math.cos(el) * Math.cos(az));
  camera.near = 0.1; camera.far = dist * 20; camera.updateProjectionMatrix(); controls.update();
}

let anim = null;
function focusOn(mesh) {
  const p = mesh.getWorldPosition(new THREE.Vector3()); const size = Math.max(mesh.userData.w, mesh.userData.d);
  const from = { t: controls.target.clone(), c: camera.position.clone() };
  const to = { t: p.clone(), c: p.clone().add(new THREE.Vector3(size * 0.6, size * 1.2, size * 1.6)) };
  const t0 = performance.now(); anim = () => {
    const k = Math.min(1, (performance.now() - t0) / 450); const e = 1 - Math.pow(1 - k, 3);
    controls.target.lerpVectors(from.t, to.t, e); camera.position.lerpVectors(from.c, to.c, e); if (k >= 1) anim = null;
  };
}

// ---------------------------------------------------------------------------------------------
// hover / info panel
// ---------------------------------------------------------------------------------------------
const fmt = (x, d = 4) => (x === undefined || x === null || Number.isNaN(x)) ? '–' : Math.abs(x) >= 1e4 || (Math.abs(x) < 1e-3 && x !== 0) ? x.toExponential(2) : x.toFixed(d);
const big = (n) => n >= 1e9 ? (n / 1e9).toFixed(2) + 'B' : n >= 1e6 ? (n / 1e6).toFixed(1) + 'M' : n >= 1e3 ? (n / 1e3).toFixed(1) + 'k' : String(n);

function showInfo(t, mesh) {
  const info = $('info'); info.style.display = 'block';
  $('iname').textContent = (t.layer !== null && t.layer !== undefined ? `block ${t.layer} · ` : '') + t.label;
  $('iid').textContent = `${t.id}   [${t.shape.join(' × ')}]`;
  const s = t.stats, d = t.delta_stats;
  const rows = [['parameters', big(t.n)], ['mean', fmt(s.mean)], ['std', fmt(s.std)], ['rms', fmt(s.rms)], ['|max|', fmt(s.absmax)]];
  if (d) rows.push(['Δ rms since prev. ckpt', fmt(d.rms)], ['Δ relative to rms', (100 * d.rel).toFixed(2) + ' %']);
  rows.push(['colour range', `${fmt(mesh.userData.lo)} … ${fmt(mesh.userData.hi)}`]);
  $('itable').innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('');
  // histogram
  const hc = $('ihist'), hx = hc.getContext('2d'); hx.clearRect(0, 0, hc.width, hc.height);
  const counts = t.hist.counts, mx = Math.max(1, ...counts), bw = hc.width / counts.length;
  hx.fillStyle = '#6fb3ff'; counts.forEach((c, i) => { const h = (c / mx) * (hc.height - 14); hx.fillRect(i * bw + 1, hc.height - 12 - h, bw - 2, h); });
  hx.fillStyle = '#8b94a7'; hx.font = '10px system-ui'; hx.fillText(fmt(t.hist.edges[0], 3), 2, hc.height - 2);
  const r = fmt(t.hist.edges[t.hist.edges.length - 1], 3); hx.fillText(r, hc.width - hx.measureText(r).width - 2, hc.height - 2);
  hx.fillText('histogram of all values', hc.width / 2 - 55, hc.height - 2);
  // tile zoom
  const tc = $('itile'), tx = tc.getContext('2d'); const img = tx.createImageData(t.cols, t.rows);
  img.data.set(paint(t, mesh.userData.lo, mesh.userData.hi));
  const off = document.createElement('canvas'); off.width = t.cols; off.height = t.rows; off.getContext('2d').putImageData(img, 0, 0);
  tc.height = t.rows === 1 ? 24 : Math.round(tc.width * t.rows / t.cols); tx.imageSmoothingEnabled = false; tx.clearRect(0, 0, tc.width, tc.height);
  tx.drawImage(off, 0, 0, tc.width, tc.height);
}

function setHover(mesh) {
  if (state.hovered === mesh) return;
  if (state.hovered) state.hovered.userData.edges.material = edgeMat;
  state.hovered = mesh;
  if (mesh) { mesh.userData.edges.material = hoverEdgeMat; showInfo(mesh.userData.tensor, mesh); }
  else $('info').style.display = 'none';
}
canvas.addEventListener('pointermove', (e) => { pointer.x = (e.clientX / window.innerWidth) * 2 - 1; pointer.y = -(e.clientY / window.innerHeight) * 2 + 1; });
canvas.addEventListener('pointerleave', () => { pointer.set(-2, -2); });
let downAt = null;
canvas.addEventListener('pointerdown', (e) => { downAt = [e.clientX, e.clientY]; });
canvas.addEventListener('pointerup', (e) => { if (downAt && Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) < 4 && state.hovered) focusOn(state.hovered); downAt = null; });
window.addEventListener('keydown', (e) => { if (e.key === 'r' && !e.target.matches('input,select')) fitCamera(); });

function frame() {
  requestAnimationFrame(frame);
  if (anim) anim();
  controls.update();
  if (state.slabs.length && pointer.x > -2) {
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObjects(state.slabs, false)[0];
    setHover(hit ? hit.object : null);
  }
  renderer.render(scene, camera);
}

// ---------------------------------------------------------------------------------------------
// legend, metrics, UI
// ---------------------------------------------------------------------------------------------
function updateLegend(gHi) {
  const c = $('legend'), ctx = c.getContext('2d'); const mi = modeInfo();
  for (let x = 0; x < c.width; x++) { const [r, g, b] = cmap(mi.name, x / (c.width - 1)); ctx.fillStyle = `rgb(${r},${g},${b})`; ctx.fillRect(x, 0, 1, c.height); }
  const global = state.scale === 'global';
  if (state.mode === 'mean') { $('lmin').textContent = global ? '-' + fmt(gHi) : '−max|mean|'; $('lmid').textContent = '0'; $('lmax').textContent = global ? '+' + fmt(gHi) : '+max|mean|'; }
  else { $('lmin').textContent = global ? '0' : 'min (per tensor)'; $('lmid').textContent = mi.label; $('lmax').textContent = global ? fmt(gHi) : 'max (per tensor)'; }
}

function updateMetrics() {
  const d = state.data; const m = d.metrics || {}; const parts = [];
  parts.push(`<b>step ${d.step}</b>${d.compare_step !== null && d.compare_step !== undefined ? ` <span>(Δ vs ${d.compare_step})</span>` : ''}`);
  if (m['train/loss'] !== undefined) parts.push(`train loss <b>${m['train/loss'].toFixed(3)}</b>`);
  if (m['val/loss'] !== undefined) parts.push(`val loss <b>${m['val/loss'].toFixed(3)}</b>`);
  if (m.tokens_seen !== undefined) parts.push(`<b>${(m.tokens_seen / 1e9).toFixed(3)}B</b> tokens`);
  if (m['perf/tokens_per_s'] !== undefined) parts.push(`${(m['perf/tokens_per_s'] / 1e3).toFixed(1)}k tok/s`);
  if (m['train/lr'] !== undefined) parts.push(`lr ${m['train/lr'].toExponential(1)}`);
  const mo = d.model;
  parts.push(`<span>${big(d.n_params)} params · L=${mo.n_layers} d=${mo.d_model} heads=${mo.n_heads}/${mo.n_kv_heads} ff=${mo.d_ff} vocab=${mo.vocab_size}${mo.tie_embeddings ? ' · tied' : ''}</span>`);
  parts.push(`<span>exported ${d.exported_at}</span>`);
  $('metrics').innerHTML = parts.join(' · ');
  $('title').textContent = d.run;
  document.title = `${d.run} @ ${d.step} — GW1B visualizer`;
}

function setData(data, keepCamera) {
  state.data = data; state.step = data.step; state.run = data.run;
  updateMetrics(); build(); if (!keepCamera) fitCamera();
  $('status').textContent = '';
  const h = $('hint'); h.textContent = data.compare_step === null || data.compare_step === undefined
    ? 'only one checkpoint so far: "change since previous checkpoint" needs two' : '';
}

async function api(path) {
  const r = await fetch(path, { cache: 'no-store' }); const j = await r.json();
  if (!r.ok || j.error) throw new Error(j.error || r.statusText); return j;
}

async function loadRuns() {
  const j = await api('/api/runs'); state.runs = j.runs;
  const sel = $('run'); const cur = state.run; sel.innerHTML = '';
  for (const r of j.runs) { const o = document.createElement('option'); o.value = r.name; o.textContent = `${r.name}  (steps ${r.steps.length}, latest ${r.latest})`; sel.appendChild(o); }
  if (cur && j.runs.some(r => r.name === cur)) sel.value = cur;
  if (!j.runs.length) { $('status').textContent = 'no runs with checkpoints found under ' + j.roots.join(', '); }
  fillSteps();
  return j.runs;
}

function fillSteps() {
  const r = state.runs.find(x => x.name === $('run').value); const sel = $('step'); sel.innerHTML = '';
  if (!r) return;
  for (const s of [...r.steps].reverse()) { const o = document.createElement('option'); o.value = s; o.textContent = s === r.latest ? `${s} (latest)` : String(s); sel.appendChild(o); }
  if (state.run === r.name && state.step !== null && r.steps.includes(state.step)) sel.value = String(state.step);
}

async function loadStep(run, step, keepCamera) {
  $('status').textContent = `loading ${run} step ${step ?? 'latest'} … (the first export of a checkpoint can take a minute)`;
  try {
    const data = await api(`/api/export?run=${encodeURIComponent(run)}&step=${step ?? ''}&compare=previous`);
    setData(data, keepCamera); fillSteps();
  } catch (e) { $('status').textContent = 'error: ' + e.message; }
}

async function poll() {   // follow latest: a new checkpoint of the current run -> reload it, camera untouched
  if (!state.follow || state.embedded) return;
  try {
    const runs = await loadRuns(); const r = runs.find(x => x.name === state.run);
    if (r && r.latest !== state.step) await loadStep(r.name, r.latest, true);
  } catch (e) { $('status').textContent = 'error: ' + e.message; }
}

function wireUI() {
  $('mode').onchange = (e) => { state.mode = e.target.value; recolor(); };
  $('scale').onchange = (e) => { state.scale = e.target.value; recolor(); };
  $('labels').onchange = (e) => { state.labels = e.target.checked; relabel(); };
  $('spacing').oninput = (e) => { state.spacing = parseFloat(e.target.value); build(); };
  $('reset').onclick = fitCamera;
  $('follow').onchange = (e) => { state.follow = e.target.checked; };
  $('run').onchange = () => { fillSteps(); loadStep($('run').value, null, false); };
  $('step').onchange = () => { loadStep($('run').value, parseInt($('step').value, 10), true); };
  $('rescan').onclick = () => loadRuns().catch(e => { $('status').textContent = 'error: ' + e.message; });
}

async function main() {
  wireUI(); frame();
  const embedded = $('gw1b-data');
  if (embedded) {
    state.embedded = true; $('runrow').style.display = 'none'; $('steprow').style.display = 'none';
    setData(JSON.parse(embedded.textContent), false); return;
  }
  try {
    const runs = await loadRuns();
    if (runs.length) await loadStep(runs[0].name, null, false);
  } catch (e) { $('status').textContent = 'error: ' + e.message; }
  state.timer = setInterval(poll, 60000);
}
main();
