import { colors, count } from './metrics.mjs';
import { readCrop } from './crop-package.mjs';

const layerNames = ['Eroded reference', 'DINOv3 RGB', 'PSPNet RGB'];
const palette = [...colors, '#172b3c'].map((hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)));
const amber = [255, 190, 100];

export function cropViewer(root, cases, benchmark, selection) {
  const get = (selector) => root.querySelector(selector);
  const file = get('[data-crop-file]');
  const controls = get('[data-local-controls]');
  const status = get('[data-local-status]');
  const caption = get('[data-local-caption]');
  const canvas = get('[data-crop-canvas]');
  let loaded, version = 0, frame = 0;

  function render() {
    if (!loaded) return;
    const { crop, planes } = loaded;
    const { model, actual, predicted } = selection();
    const area = crop.width * crop.height;
    const paired = ['dinov3', 'pspnet'].includes(model) && actual !== predicted;
    const modelLayer = model === 'dinov3' ? 1 : 2;
    const read = (name) => Number(get(`[name=${name}]`).value);
    const left = read('layer-left'), right = read('layer-right');
    const swipe = Math.round(canvas.width * read('swipe') / 100);
    const zoom = read('zoom');
    const opacity = read('opacity') / 100;
    const ctx = canvas.getContext('2d');
    const rgba = ctx.createImageData(canvas.width, canvas.height);
    let countPair = 0, nearIgnored = 0;
    if (paired) {
      for (let i = 0; i < area; i++) {
        if (planes[i] !== actual || planes[i + modelLayer * area] !== predicted) continue;
        countPair++;
        const x = i % crop.width, y = Math.floor(i / crop.width);
        if ((x > 0 && planes[i - 1] === 6) || (x + 1 < crop.width && planes[i + 1] === 6) ||
            (y > 0 && planes[i - crop.width] === 6) || (y + 1 < crop.height && planes[i + crop.width] === 6)) nearIgnored++;
      }
    }
    for (let y = 0; y < canvas.height; y++) {
      for (let x = 0; x < canvas.width; x++) {
        const sx = Math.floor(crop.width * (.5 + ((x + .5) / canvas.width - .5) / zoom));
        const sy = Math.floor(crop.height * (.5 + ((y + .5) / canvas.height - .5) / zoom));
        const source = sy * crop.width + sx;
        const layer = x < swipe ? left : right;
        const rgb = palette[planes[source + layer * area]];
        const divider = x === swipe && swipe > 0;
        const marked = paired && planes[source] === actual && planes[source + modelLayer * area] === predicted;
        const j = 4 * (y * canvas.width + x);
        for (let ch = 0; ch < 3; ch++) rgba.data[j + ch] = divider ? 246 : marked ? rgb[ch] * (1 - opacity) + amber[ch] * opacity : rgb[ch];
        rgba.data[j + 3] = 255;
      }
    }
    ctx.putImageData(rgba, 0, 0);
    const prefix = `Potsdam ${crop.scene} · row ${crop.row}, column ${crop.col} · ${layerNames[left]} | ${layerNames[right]}`;
    caption.textContent = paired ? `${prefix} · ${count(countPair)} matching pixels; ${count(nearIgnored)} touch the ignored mask (4-neighbor).` : prefix;
  }

  file.addEventListener('change', async () => {
    const current = ++version;
    loaded = null;
    controls.hidden = true;
    if (!file.files[0]) return;
    status.textContent = 'Checking geometry, provenance and class counts…';
    try {
      const result = await readCrop(file.files[0], cases, benchmark);
      if (current !== version) return;
      loaded = result;
      controls.hidden = false;
      status.textContent = 'Local crop reproduces the published joint counts.';
      render();
    } catch (error) {
      if (current === version) status.textContent = `Crop could not be opened: ${error.message}`;
    }
  });
  controls.addEventListener('input', () => {
    if (!frame) frame = requestAnimationFrame(() => { frame = 0; render(); });
  });
  get('[data-save-crop]').addEventListener('click', () => {
    const { crop } = loaded;
    const saved = document.createElement('canvas');
    saved.width = canvas.width; saved.height = canvas.height + 54;
    const context = saved.getContext('2d');
    context.drawImage(canvas, 0, 0);
    context.fillStyle = '#0b2239';
    context.fillRect(0, canvas.height, canvas.width, 54);
    context.fillStyle = 'white';
    context.font = '13px monospace';
    context.fillText(`Potsdam ${crop.scene} · eroded reference · row ${crop.row}, col ${crop.col}`, 8, canvas.height + 21);
    context.fillText(`${layerNames[get('[name=layer-left]').value]} | ${layerNames[get('[name=layer-right]').value]} · local paired crop`, 8, canvas.height + 42);
    const link = document.createElement('a');
    link.href = saved.toDataURL('image/png');
    link.download = `potsdam-${crop.scene}-${crop.row}-${crop.col}-comparison.png`;
    link.click();
  });
  return render;
}
