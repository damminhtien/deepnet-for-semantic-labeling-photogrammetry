import evidence from '../generated/evidence.json';
import { count } from './metrics.mjs';
import { restore, share } from './state.mjs';

const root = document.querySelector('[data-atlas]');
const get = (selector) => root.querySelector(selector);
const dataset = get('[name=dataset]');
const scene = get('[name=scene]');
const mapping = get('[name=mapping]');
const band = get('[name=band]');
const zoom = get('[name=zoom]');
const swipe = get('[name=swipe]');
const canvas = get('[data-raster]');
const context = canvas.getContext('2d');
const fileInput = get('[data-tiff]');
const status = get('[data-status]');
const rgbir = evidence.spectral.bands;
const vaihingen = ['ir', 'red', 'green'];
const names = { red: 'Red', green: 'Green', blue: 'Blue', ir: 'Near infrared' };
const sceneEntries = Object.entries(evidence.spectral.scenes);
const xs = sceneEntries.flatMap(([, { extent }]) => [extent[0], extent[2]]);
const ys = sceneEntries.flatMap(([, { extent }]) => [extent[1], extent[3]]);
const minX = Math.min(...xs), maxY = Math.max(...ys);
const scale = Math.min(510 / (Math.max(...xs) - minX), 288 / (maxY - Math.min(...ys)));

let image = null;
let fileVersion = 0, renderVersion = 0, probeVersion = 0;
let center = [0, 0], pixel = [0, 0], viewport = null, raster = null;

function isPotsdam() { return dataset.value === 'Potsdam'; }
function bands() {
  if (!isPotsdam()) return mapping.value === 'nir' ? [1] : [1, 2, 3];
  const historical = evidence.spectral.scenes[scene.value].historical_bands;
  return { historical, nominal: [4, 1, 2], compare: [4, 2, 3, 1], rgb: [1, 2, 3], nir: [4] }[mapping.value];
}

function drawMap() {
  const map = get('[data-map]');
  map.innerHTML = sceneEntries.map(([key, { extent: [left, bottom, right, top] }]) => {
    const x = 24 + (left - minX) * scale, y = 20 + (maxY - top) * scale;
    const w = (right - left) * scale, h = (top - bottom) * scale;
    const selected = scene.value === key;
    return `<g role="button" tabindex="0" data-tile="${key}" aria-label="Scene ${key}" aria-pressed="${selected}"><rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${selected ? '#9b6137' : '#26475d'}" stroke="${selected ? '#f1bb83' : '#567688'}"/><text x="${x + w / 2}" y="${y + h / 2 + 3}" font-size="9" fill="white" text-anchor="middle">${key}</text></g>`;
  }).join('');
  get('[data-scene-meta]').textContent = `Scene ${scene.value} · ${evidence.spectral.scenes[scene.value].shape.join(' × ')} pixels · 5 cm ground sampling distance`;
}

function histogram() {
  const values = evidence.histograms[dataset.value][band.value === 'ir' ? 'nir' : band.value];
  const svg = get('[data-histogram]');
  const peak = Math.max(...Object.values(evidence.histograms[dataset.value]).flat());
  const points = values.map((value, i) => `${(i * 420 / 255).toFixed(2)},${(122 - 115 * value / peak).toFixed(2)}`);
  svg.innerHTML = `<path d="M0 122H420M210 6V122" stroke="#567688" stroke-dasharray="3 5"/><path d="M0,122 L${points.join(' ')} L420,122Z" fill="#77acb8" fill-opacity=".38" stroke="#a6d3d8"/>`;
  svg.setAttribute('aria-label', `${dataset.value} training histogram for ${names[band.value]}`);
  get('[data-hist-note]').textContent = `${names[band.value]} · ${count(values.reduce((a, b) => a + b, 0))} training pixels. Vertical scale stays fixed across bands in this dataset.`;
}

function update() {
  const potsdam = isPotsdam();
  get('[data-footprint]').hidden = !potsdam;
  get('[data-scene-control]').hidden = !potsdam;
  for (const option of mapping.options) option.disabled = !potsdam && ['nominal', 'rgb', 'compare'].includes(option.value);
  if (mapping.selectedOptions[0].disabled) mapping.value = 'historical';
  const sourceBands = potsdam ? rgbir : vaihingen;
  const indices = bands();
  const currentBand = band.value;
  const compare = mapping.value === 'compare';
  get('[data-spectral-swipe]').hidden = !compare;
  const available = indices.map((index) => sourceBands[index - 1]);
  band.replaceChildren(...available.map((key) => new Option(names[key], key)));
  band.value = available.includes(currentBand) ? currentBand : available[0];
  const table = get('[data-band-table]');
  table.replaceChildren(...indices.map((index, i) => {
    const row = document.createElement('tr');
    const display = compare ? 'Comparison source' : indices.length === 1 ? 'Grayscale' : ['Red', 'Green', 'Blue'][i];
    for (const value of [display, index, names[sourceBands[index - 1]]]) row.insertCell().textContent = value;
    return row;
  }));
  get('[data-finding]').textContent = potsdam
    ? 'The audited historical files map to RGBIR source bands 4 / 2 / 3 (NIR / green / blue). The historical names imply 4 / 1 / 2 (NIR / red / green). Compare the two interpretations.'
    : 'The Vaihingen three-band source encodes NIR / red / green. Its local catalog is pixel-local, so no projected footprint is shown.';
  get('[data-file-hint]').textContent = potsdam
    ? 'Choose the official four-band RGBIR TIFF for the selected scene. The Atlas verifies its SHA-256. Historical mode displays the audited band mapping from RGBIR; it does not reproduce historical digital numbers.'
    : 'Choose an official 8-bit, three-band Vaihingen TIFF. Values are shown from your file; the Atlas does not verify its identity.';
  if (potsdam) drawMap();
  histogram();
}

async function render() {
  if (!image) return;
  const version = ++renderVersion;
  const width = image.getWidth(), height = image.getHeight();
  const ratio = Number(zoom.value);
  const w = Math.ceil(width / ratio), h = Math.ceil(height / ratio);
  const x = Math.min(width - w, Math.max(0, Math.round(center[0] - w / 2)));
  const y = Math.min(height - h, Math.max(0, Math.round(center[1] - h / 2)));
  const outputWidth = Math.min(640, w), outputHeight = Math.max(1, Math.round(outputWidth * h / w));
  const window = [x, y, w, h];
  const key = window.join(',');
  try {
    if (raster?.key !== key) {
      const values = await image.readRasters({ window: [x, y, x + w, y + h], width: outputWidth,
        height: outputHeight, resampleMethod: 'nearest' });
      if (version !== renderVersion) return;
      raster = { key, values, outputWidth, outputHeight };
    }
    viewport = window;
    paint();
  } catch {
    if (version === renderVersion) status.textContent = 'Could not decode this TIFF window.';
  }
}

function paint() {
  if (!raster || !viewport) return;
  const { values, outputWidth, outputHeight } = raster;
  if (canvas.width !== outputWidth) canvas.width = outputWidth;
  if (canvas.height !== outputHeight) canvas.height = outputHeight;
  const compare = mapping.value === 'compare';
  const historical = isPotsdam() && evidence.spectral.scenes[scene.value].historical_bands;
  const indices = compare ? historical : bands();
  const implied = [4, 1, 2];
  const split = Math.round(outputWidth * Number(swipe.value) / 100);
  const rgba = context.createImageData(outputWidth, outputHeight);
  for (let i = 0; i < outputWidth * outputHeight; i++) {
    const column = i % outputWidth;
    const current = compare && column >= split ? implied : indices;
    const divider = compare && column === split && split > 0;
    for (let ch = 0; ch < 3; ch++) {
      rgba.data[4 * i + ch] = divider ? 245 : values[current[current.length === 1 ? 0 : ch] - 1][i];
    }
    rgba.data[4 * i + 3] = 255;
  }
  context.putImageData(rgba, 0, 0);
  const [x, y, w, h] = viewport;
  if (pixel[0] >= x && pixel[0] < x + w && pixel[1] >= y && pixel[1] < y + h) {
    const px = (pixel[0] - x + .5) * outputWidth / w;
    const py = (pixel[1] - y + .5) * outputHeight / h;
    context.strokeStyle = '#ffffff';
    context.lineWidth = 2;
    context.strokeRect(px - 5, py - 5, 10, 10);
  }
}

async function probe() {
  if (!image) return;
  const version = ++probeVersion;
  const [x, y] = pixel;
  const sourceBands = isPotsdam() ? rgbir : vaihingen;
  try {
    const values = await image.readRasters({ window: [x, y, x + 1, y + 1], interleave: true });
    if (version !== probeVersion) return;
    get('[data-pixel]').textContent = `Column ${x} · row ${y} / ${Array.from(values, (value, i) => `B${i + 1} ${names[sourceBands[i]]}: ${value}`).join(' · ')} DN`;
  } catch {
    if (version === probeVersion) get('[data-pixel]').textContent = 'Pixel could not be read.';
  }
}

function reset() {
  ++fileVersion; ++renderVersion; ++probeVersion;
  image = viewport = raster = null;
  fileInput.value = '';
  get('[data-viewer]').hidden = true;
  status.textContent = 'Files stay on this device.';
}

fileInput.addEventListener('change', async () => {
  const file = fileInput.files[0];
  if (!file) return;
  const version = ++fileVersion;
  ++renderVersion; ++probeVersion;
  image = viewport = raster = null;
  get('[data-viewer]').hidden = true;
  status.textContent = 'Reading local TIFF…';
  try {
    const potsdam = isPotsdam();
    if (potsdam && file.size > 250 * 1024 * 1024) throw new Error('This file is too large to match the audited source.');
    if (potsdam) {
      const bytes = new Uint8Array(await crypto.subtle.digest('SHA-256', await file.arrayBuffer()));
      const digest = Array.from(bytes, (value) => value.toString(16).padStart(2, '0')).join('');
      if (digest !== evidence.spectral.scenes[scene.value].sha256) throw new Error('SHA-256 does not match the selected Potsdam scene.');
    }
    const { fromBlob } = await import('geotiff');
    const next = await (await fromBlob(file)).getImage();
    if (version !== fileVersion) return;
    if (next.getSamplesPerPixel() !== (potsdam ? 4 : 3) ||
        next.getFileDirectory().getValue('BitsPerSample').some((value) => value !== 8)) {
      throw new Error('Choose an 8-bit TIFF with the expected band count.');
    }
    image = next;
    const identity = potsdam ? 'SHA-256 verified against the public scene audit.' : 'Local source · identity not verified.';
    const size = [image.getWidth(), image.getHeight()];
    const params = new URLSearchParams(location.search);
    const fromUrl = (name, i) => {
      const value = params.get(name);
      return value !== null && /^\d+$/.test(value) && Number(value) < size[i] ? Number(value) : Math.floor(size[i] / 2);
    };
    pixel = [fromUrl('x', 0), fromUrl('y', 1)];
    center = [params.has('cx') ? fromUrl('cx', 0) : pixel[0], params.has('cy') ? fromUrl('cy', 1) : pixel[1]];
    get('[data-viewer]').hidden = false;
    status.textContent = identity;
    render();
    probe();
  } catch (error) {
    if (version === fileVersion) status.textContent = error.message || 'Could not open this TIFF.';
  }
});

root.addEventListener('change', (event) => {
  if (event.target === fileInput) return;
  if (event.target === dataset || event.target === scene) {
    reset();
    share(root, { x: null, y: null, cx: null, cy: null });
  }
  update();
  share(root);
  render();
});
get('[data-map]').addEventListener('click', (event) => selectTile(event.target.closest('[data-tile]')));
get('[data-map]').addEventListener('keydown', (event) => {
  if (event.key === ' ' || event.key === 'Enter') {
    event.preventDefault();
    const tile = event.target.closest('[data-tile]');
    selectTile(tile);
    if (tile) get(`[data-tile="${tile.dataset.tile}"]`).focus();
  }
});
function selectTile(tile) {
  if (!tile) return;
  scene.value = tile.dataset.tile;
  scene.dispatchEvent(new Event('change', { bubbles: true }));
}
function selectPixel(x, y) {
  pixel = [Math.max(0, Math.min(image.getWidth() - 1, x)), Math.max(0, Math.min(image.getHeight() - 1, y))];
  share(root, { x: pixel[0], y: pixel[1] });
  render();
  probe();
}
canvas.addEventListener('click', (event) => {
  if (!viewport) return;
  const box = canvas.getBoundingClientRect();
  selectPixel(viewport[0] + Math.floor((event.clientX - box.left) / box.width * viewport[2]),
    viewport[1] + Math.floor((event.clientY - box.top) / box.height * viewport[3]));
});
canvas.addEventListener('keydown', (event) => {
  const direction = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[event.key];
  if (!image || !direction) return;
  event.preventDefault();
  selectPixel(pixel[0] + direction[0], pixel[1] + direction[1]);
});
swipe.addEventListener('input', paint);
get('[data-center]').addEventListener('click', () => {
  center = [...pixel];
  share(root, { cx: center[0], cy: center[1] });
  render();
});
restore(root);
update();
restore(root);
histogram();
