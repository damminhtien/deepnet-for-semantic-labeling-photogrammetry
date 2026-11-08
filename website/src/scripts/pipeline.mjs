import { colors, percent, shortClasses } from './metrics.mjs';
import { route } from './routes.mjs';
import { restore, share } from './state.mjs';

const root = document.querySelector('[data-pipeline]');
const get = (selector) => root.querySelector(selector);
const palette = colors.map((hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)));
const ignored = [215, 155, 98];

function asset(view) {
  const img = new Image();
  img.src = `${route()}schematic/${view}.svg`;
  return img.decode().then(() => img);
}

const ready = Promise.all([asset('image'), asset('labels'), asset('ids')]);
restore(root);
ready.then(([optical, semantic, indexed]) => {
  const source = document.createElement('canvas');
  const size = source.width = source.height = indexed.width;
  const ctx = source.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(indexed, 0, 0);
  const rgba = ctx.getImageData(0, 0, size, size).data;
  const ids = Uint8Array.from({ length: size * size }, (_, i) => Math.max(0, Math.min(5, Math.round((rgba[i * 4] - 30) / 40))));
  const rare = [[], []];
  ids.forEach((id, i) => { if (id >= 4) rare[id - 4].push(i); });

  function sample() {
    const schedule = get('[name=schedule]').value;
    let seed = Number(get('[name=draw]').value);
    const rand = () => (seed = (Math.imul(1664525, seed) + 1013904223) >>> 0) / 2 ** 32;
    const patch = 112;
    const windows = Array.from({ length: 24 }, (_, i) => [Math.round((i % 6) * (size - patch) / 5), Math.round(Math.floor(i / 6) * (size - patch) / 3)]);
    if (schedule === 'rare') {
      const clamp = (n) => Math.max(0, Math.min(size - patch, n));
      for (const points of rare) {
        for (let i = 0; i < 4; i++) {
          const index = points[Math.floor(rand() * points.length)];
          windows.push([clamp(index % size - patch / 2), clamp(Math.floor(index / size) - patch / 2)]);
        }
      }
    }
    const output = get('[data-sample]').getContext('2d');
    output.drawImage(semantic, 0, 0);
    const exposure = Array(6).fill(0);
    windows.forEach(([x, y], i) => {
      for (let row = y; row < y + patch; row++) {
        for (let col = x; col < x + patch; col++) exposure[ids[row * size + col]]++;
      }
      output.strokeStyle = i < 24 ? '#f2efe5aa' : '#ffcb8c';
      output.lineWidth = i < 24 ? 1 : 3;
      output.strokeRect(x + 1, y + 1, patch - 2, patch - 2);
    });
    get('[data-windows]').textContent = `${windows.length} total windows · ${windows.length - 24} extra rare-class draws`;
    get('[data-exposure]').innerHTML = exposure.map((n, i) => {
      const fraction = n / (patch * patch * windows.length);
      return `<div class="bar-row"><span>${shortClasses[i]}</span><strong>${percent(fraction, 1)}</strong><div><i style="--value:${fraction * 100}%;background:${colors[i]}"></i></div></div>`;
    }).join('');
  }

  function augment() {
    const values = ['rotation', 'scale', 'translation', 'gain'].map((key, i) => {
      const value = Number(get(`[name=${key}]`).value);
      get(`[for=${key}]`).value = value + (i === 0 ? '°' : '%');
      return value;
    });
    const [angle, scale, shift, gain] = [values[0] * Math.PI / 180, ...values.slice(1).map((value) => value / 100)];
    const output = get('[data-image]').getContext('2d', { willReadFrequently: true });
    const labels = get('[data-labels]').getContext('2d');
    const valid = get('[data-valid]').getContext('2d');
    const n = output.canvas.width;
    output.clearRect(0, 0, n, n);
    output.save();
    output.translate(n * (.5 + shift), n / 2);
    output.rotate(angle);
    output.scale(scale, scale);
    output.drawImage(optical, -n / 2, -n / 2, n, n);
    output.restore();
    const imageData = output.getImageData(0, 0, n, n);
    const labelData = labels.createImageData(n, n);
    const validData = valid.createImageData(n, n);
    const c = Math.cos(angle), s = Math.sin(angle);
    let invalid = 0;
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const dx = x + .5 - n * (.5 + shift), dy = y + .5 - n / 2;
        const u = Math.floor(size * (.5 + (c * dx + s * dy) / (n * scale)));
        const v = Math.floor(size * (.5 + (-s * dx + c * dy) / (n * scale)));
        const inside = u >= 0 && u < size && v >= 0 && v < size;
        if (!inside) invalid++;
        const i = 4 * (y * n + x);
        labelData.data.set([...(inside ? palette[ids[v * size + u]] : ignored), 255], i);
        validData.data.set([...(inside ? [34, 68, 87] : ignored), 255], i);
        for (let ch = 0; ch < 3; ch++) imageData.data[i + ch] = inside ? imageData.data[i + ch] * gain : [17, 44, 62][ch];
        imageData.data[i + 3] = 255;
      }
    }
    output.putImageData(imageData, 0, 0);
    labels.putImageData(labelData, 0, 0);
    valid.putImageData(validData, 0, 0);
    get('[data-invalid]').textContent = `${percent(invalid / (n * n), 1)} invalid after transform · ${invalid.toLocaleString('en-US')} pixels assigned ignore ID 6`;
  }

  root.addEventListener('input', (event) => {
    if (event.target.type === 'range') augment();
  });
  root.addEventListener('change', () => { share(root); sample(); });
  get('[data-reset]').addEventListener('click', () => {
    for (const slider of root.querySelectorAll('input[type=range]')) slider.value = slider.defaultValue;
    share(root);
    augment();
  });
  sample();
  augment();
}).catch(() => { get('[data-windows]').textContent = 'The schematic could not be loaded.'; });
