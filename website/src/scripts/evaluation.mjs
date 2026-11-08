import evidence from '../generated/evidence.json';
import { count, cropPixels, metrics, percent, shortClasses } from './metrics.mjs';
import { source, route } from './routes.mjs';
import { query, restore, share } from './state.mjs';
import { cropViewer } from './crop-viewer.mjs';

const root = document.querySelector('[data-evaluation]');
const get = (selector) => root.querySelector(selector);
const model = get('[name=model]');
const scene = get('[name=scene]');
const units = get('[name=units]');
const { benchmark, cases } = evidence;
const names = benchmark.class_names;
const params = query();
function classId(key, fallback) {
  const raw = params.get(key);
  return raw !== null && /^[0-5]$/.test(raw) ? Number(raw) : fallback;
}
let actual = classId('truth', 2), predicted = classId('pred', 3);
let openCrop = params.get('crop');
const cropId = (crop) => `${crop.scene}-${crop.row}-${crop.col}`;
const pairedModel = () => ['dinov3', 'pspnet'].includes(model.value);
const claim = () => ({ dinov3: '001', pspnet: '002', dinov2: '003', sam: '004' }[model.value]);
const updateURL = () => share(root, { truth: actual, pred: predicted, crop: openCrop });
const renderLocal = cropViewer(root, cases, benchmark, () => ({ model: model.value, actual, predicted }));

function renderCrops() {
  const list = get('[data-crops]');
  if (!pairedModel()) {
    list.textContent = 'Paired crop measurements are available for DINOv3 RGB and PSPNet RGB.';
    return;
  }
  if (actual === predicted) {
    list.textContent = 'Select an off-diagonal cell to explore a documented error.';
    return;
  }
  const found = cases
    .filter((crop) => scene.value === 'all' || crop.scene === scene.value)
    .map((crop) => ({ crop, n: cropPixels(crop, model.value, actual, predicted) }))
    .filter(({ n }) => n > 0)
    .sort((a, b) => b.n - a.n);
  if (!found.length) {
    list.textContent = 'This pair has no matching crop in the published gallery.';
    return;
  }
  list.replaceChildren(...found.map(({ crop, n }) => {
    const details = document.createElement('details');
    details.open = cropId(crop) === openCrop;
    const label = document.createElement('summary');
    label.textContent = `Scene ${crop.scene} · ${count(n)} matching pixels`;
    const body = document.createElement('div');
    const report = source(`docs/research/figures/paired-cases-${crop.case_group}.png`);
    body.innerHTML = `<p class="note">Row ${crop.row}, column ${crop.col} · ${crop.height} × ${crop.width} window</p>
      <p class="note">${crop.case_group_title} · ${percent(n / crop.valid_pixels)} of the scored crop</p>
      <p><a href="${report}">View published case figure ↗</a> · <button type="button" data-download>Export evidence</button></p>`;
    body.querySelector('button').addEventListener('click', () => {
      const { transitions, ...window } = crop;
      const content = {
        window, selection: { reference: names[actual], prediction: names[predicted], pixels: n },
        model: benchmark.models[model.value].experiment_id,
        checkpoint_sha256: benchmark.models[model.value].checkpoint_sha256,
        dataset: benchmark.dataset_id, split_sha256: benchmark.split_sha256,
        reference: benchmark.label_protocol, class_order: names,
        transitions, published_figure: report,
      };
      const url = URL.createObjectURL(new Blob([JSON.stringify(content, null, 2)], { type: 'application/json' }));
      const link = document.createElement('a');
      link.href = url;
      link.download = `potsdam-${cropId(crop)}-${model.value}.json`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
    details.append(label, body);
    label.addEventListener('click', (event) => {
      event.preventDefault();
      const wasOpen = details.open;
      for (const other of list.querySelectorAll('details')) other.open = false;
      details.open = !wasOpen;
      openCrop = details.open ? cropId(crop) : null;
      updateURL();
    });
    return details;
  }));
}

function render() {
  const selected = benchmark.models[model.value];
  const matrix = scene.value === 'all' ? selected.aggregate_confusion_matrix
    : selected.scenes.find((item) => item.scene_id === scene.value).confusion_matrix;
  const { rows, total, errors, oa, miou, iou } = metrics(matrix);
  const labels = ['Mean IoU', 'Overall accuracy', 'Scored pixels'];
  const values = [percent(miou), percent(oa), count(total)];
  get('[data-scores]').innerHTML = labels.map((label, i) => `<span><strong>${values[i]}</strong>${label}</span>`).join('');
  get('[data-head]').innerHTML = `<tr><th></th>${shortClasses.map((label) => `<th scope="col">${label}</th>`).join('')}</tr>`;
  const highest = Math.max(...selected.aggregate_confusion_matrix.flat());
  get('[data-matrix]').innerHTML = matrix.map((row, r) => `<tr><th scope="row">${shortClasses[r]}</th>${row.map((n, c) => {
    const fraction = rows[r] ? n / rows[r] : 0;
    const heat = units.value === 'row' ? Math.sqrt(fraction) : Math.log1p(n) / Math.log1p(highest);
    const visible = units.value === 'row' ? (rows[r] ? percent(fraction, 1) : '—') : count(n);
    return `<td><button type="button" data-pair-button="${r},${c}" aria-pressed="${r === actual && c === predicted}" aria-label="${names[r]} to ${names[c]}: ${count(n)} pixels" style="--heat:${.08 + .55 * heat}">${visible}</button></td>`;
  }).join('')}</tr>`).join('');
  const n = matrix[actual][predicted];
  get('[data-pair]').textContent = `${names[actual]} → ${names[predicted]}`;
  get('[data-pair-count]').textContent = actual === predicted
    ? `${count(n)} correct pixels · ${percent(iou[actual])} class IoU`
    : `${count(n)} pixels · ${errors ? percent(n / errors) : '0%'} of errors in this view`;

  const sceneList = get('[data-scenes]');
  if (scene.value === 'all') {
    const leaders = selected.scenes.map((item) => [item.scene_id, item.confusion_matrix[actual][predicted]])
      .filter(([, pixels]) => pixels > 0).sort((a, b) => b[1] - a[1]).slice(0, 4);
    sceneList.innerHTML = leaders.length
      ? `<p class="note">Highest counts by scene</p><div class="controls">${leaders.map(([id, pixels]) => `<button type="button" data-scene="${id}">${id} · ${count(pixels)}</button>`).join('')}</div>`
      : '<p class="note">No scene has this class pair.</p>';
  } else {
    sceneList.innerHTML = '<button type="button" data-scene="all">Return to all scenes</button>';
  }
  get('[data-model-note]').innerHTML = `${pairedModel() ? 'Matched RGB evaluation setup.' : 'Historical recipe; its preprocessing or inference differs from the matched RGB pair.'} <a href="${route('methods')}#claim-${claim()}">Model provenance ↗</a>`;
  renderCrops();
  renderLocal();
}

get('[data-matrix]').addEventListener('click', (event) => {
  const button = event.target.closest('[data-pair-button]');
  if (!button) return;
  const key = button.dataset.pairButton;
  [actual, predicted] = key.split(',').map(Number);
  openCrop = null;
  render(); updateURL();
  get(`[data-pair-button="${key}"]`).focus({ preventScroll: true });
});
get('[data-scenes]').addEventListener('click', (event) => {
  const button = event.target.closest('[data-scene]');
  if (button) {
    scene.value = button.dataset.scene;
    scene.dispatchEvent(new Event('change', { bubbles: true }));
    scene.focus({ preventScroll: true });
  }
});
root.addEventListener('change', (event) => {
  if (event.target.type === 'file') return;
  if ([model, scene].includes(event.target)) openCrop = null;
  if ([model, scene, units].includes(event.target)) render();
  updateURL();
});
restore(root);
render();
