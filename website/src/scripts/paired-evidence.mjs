import manifest from '../generated/visual-evidence.json';
import { restore, share } from './state.mjs';
import { source } from './routes.mjs';

const instances = new WeakMap();
const imageCache = new Map();
const cases = manifest.case_studies;
const assets = new Map(manifest.assets.map((asset) => [asset.asset_id, asset]));
const layerNames = {
  reference: 'Eroded reference',
  dinov3: 'DINOv3 RGB prediction',
  pspnet: 'PSPNet RGB prediction',
  paired_outcome: 'Paired correctness outcome',
};

function figureImage(root, asset) {
  const url = root.dataset.baseUrl + '/' + asset.file;
  if (!imageCache.has(url)) {
    imageCache.set(url, new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error('Could not load archived paired figure: ' + asset.file));
      image.src = url;
    }));
  }
  return imageCache.get(url);
}

function init(element) {
  if (instances.has(element)) return instances.get(element);
  const baseCanvas = element.querySelector('[data-base]');
  const overlayCanvas = element.querySelector('[data-overlay]');
  const viewport = element.querySelector('[data-viewport]');
  const fallback = element.querySelector('[data-figure-fallback]');
  const caseGroup = element.querySelector('[data-case-group]');
  const caseSelect = element.querySelector('[data-case-select]');
  const modelSelect = element.querySelector('[data-model]');
  const layerSelect = element.querySelector('[data-layer]');
  const swipe = element.querySelector('[data-swipe]');
  const opacity = element.querySelector('[data-opacity]');
  const zoom = element.querySelector('[data-zoom]');
  const status = element.querySelector('[data-paired-status]');
  const caption = element.querySelector('[data-case-caption]');
  const detail = element.querySelector('[data-case-detail]');
  const modelProvenance = element.querySelector('[data-model-provenance]');
  const figureSize = element.querySelector('[data-source-size]');
  const figureLink = element.querySelector('.paired-caption a');
  const outcomeLegend = element.querySelector('[data-outcome-legend]');
  const classLegend = element.querySelector('[data-class-legend]');
  let selectedModel = element.dataset.initialModel || 'dinov3';
  let selectedLayer = 'prediction';
  let selectedCase = null;
  let selectedContext = null;
  let panX = 0;
  let panY = 0;
  let drag = null;

  function groupsFor(id) {
    return cases.filter((item) => item.figure_asset_id === id);
  }

  function fillCases(groupId, preferredId) {
    if (!caseSelect) return;
    const options = groupsFor(groupId).map((item) => {
      const option = document.createElement('option');
      option.value = item.case_id;
      const box = item.pixel_window;
      option.textContent = item.scene_id + ' · r' + box.row + ', c' + box.col;
      return option;
    });
    caseSelect.replaceChildren(...options);
    if (options.length) {
      caseSelect.value = options.some((option) => option.value === preferredId)
        ? preferredId
        : options[0].value;
    }
  }

  function clearCanvases() {
    for (const canvas of [baseCanvas, overlayCanvas]) {
      canvas?.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
    }
    element.dataset.ready = 'false';
    element.dataset.unavailable = 'true';
  }

  function selectedPanel() {
    if (selectedLayer === 'prediction') return selectedModel;
    return selectedLayer;
  }

  function updateTransform() {
    if (!viewport || !zoom) return;
    const factor = Number(zoom.value) || 1;
    const translate = 'translate(' + panX + 'px, ' + panY + 'px) scale(' + factor + ')';
    baseCanvas.style.transform = translate;
    overlayCanvas.style.transform = translate;
    viewport.dataset.zoomed = String(factor > 1);
    if (factor === 1) panX = panY = 0;
  }

  async function draw() {
    if (!selectedCase || !selectedModel || !selectedLayer) return;
    const asset = assets.get(selectedCase.figure_asset_id);
    const baseCrop = selectedCase.panels.rgb;
    const overlayCrop = selectedCase.panels[selectedPanel()];
    if (!asset || !baseCrop || !overlayCrop) return;
    try {
      const image = await figureImage(element, asset);
      for (const [canvas, crop] of [[baseCanvas, baseCrop], [overlayCanvas, overlayCrop]]) {
        canvas.width = crop.width;
        canvas.height = crop.height;
        const context = canvas.getContext('2d', { alpha: false });
        context.clearRect(0, 0, crop.width, crop.height);
        context.imageSmoothingEnabled = false;
        context.drawImage(image, crop.x, crop.y, crop.width, crop.height, 0, 0, crop.width, crop.height);
      }
      element.dataset.ready = 'true';
      element.dataset.unavailable = 'false';
      fallback.alt = 'Archived paired case figure, scene ' + selectedCase.scene_id + ', crop row ' + selectedCase.pixel_window.row + ', column ' + selectedCase.pixel_window.col + '. Comparison shown: RGB and ' + layerNames[selectedPanel()] + '.';
      caption.textContent = selectedCase.case_group_title + ' · Scene ' + selectedCase.scene_id;
      const crop = selectedCase.pixel_window;
      const position = 'Row ' + crop.row + ', column ' + crop.col + ' · ' + crop.height + ' × ' + crop.width + ' source pixels';
      const support = selectedContext?.matchingPixels
        ? ' · ' + selectedContext.matchingPixels.toLocaleString('en-US') + ' matching pixels in this crop'
        : '';
      detail.textContent = position + support;
      modelProvenance.textContent = ['dinov3', 'pspnet'].map((name) => {
        const model = selectedCase.models[name];
        return model.model_id + ' · SHA-256 ' + model.checkpoint_sha256.slice(0, 12);
      }).join(' / ');
      figureSize.textContent = asset.width_px + ' × ' + asset.height_px + ' px source figure';
      figureLink.href = source(asset.source_artifact_path);
      element.style.setProperty('--split', (Number(swipe.value) || 52) + '%');
      element.style.setProperty('--overlay-opacity', String((Number(opacity.value) || 0) / 100));
      status.textContent = selectedContext?.matchingPixels
        ? 'The selected crop contains ' + selectedContext.matchingPixels.toLocaleString('en-US') + ' pixels matching this confusion pair. The published figure does not mark their exact locations.'
        : 'Archived figure panel crops; the swipe compares the same scene window across recorded layers.';
      outcomeLegend.hidden = selectedLayer !== 'paired_outcome';
      classLegend.hidden = selectedLayer === 'paired_outcome';
      viewport.setAttribute('aria-label', 'Scene ' + selectedCase.scene_id + '. Left layer: RGB source rendering. Right layer: ' + layerNames[selectedPanel()] + '. Drag to pan after zooming; arrow keys pan when focused.');
      updateTransform();
    } catch (error) {
      clearCanvases();
      status.textContent = error.message;
    }
  }

  function setCase(record, context = null) {
    const found = typeof record === 'string'
      ? cases.find((item) => item.case_id === record)
      : cases.find((item) => {
          if (record?.case_id) return item.case_id === record.case_id;
          const window = record?.pixel_window ?? record;
          return item.figure_asset_id === (record?.figure_asset_id ?? record?.case_group)
            && item.scene_id === (record?.scene_id ?? record?.scene)
            && item.pixel_window.row === (window?.row ?? record?.row)
            && item.pixel_window.col === (window?.col ?? record?.col);
        });
    if (!found) return false;
    selectedCase = found;
    selectedContext = context;
    if (caseGroup) caseGroup.value = found.figure_asset_id;
    if (caseSelect) {
      fillCases(found.figure_asset_id, found.case_id);
      caseSelect.value = found.case_id;
    }
    const asset = assets.get(found.figure_asset_id);
    fallback.src = element.dataset.baseUrl + '/' + asset.file;
    fallback.width = asset.width_px;
    fallback.height = asset.height_px;
    fallback.alt = 'Published paired case figure for ' + found.case_group_title + ', scene ' + found.scene_id + '.';
    if (figureLink) figureLink.href = source(asset.source_artifact_path);
    if (selectedContext?.publicCase) {
      const url = new URL(location.href);
      url.searchParams.set('public-case', found.case_id);
      history.replaceState(null, '', url);
    }
    draw();
    return true;
  }

  function setModel(value) {
    if (!['dinov3', 'pspnet'].includes(value)) return false;
    selectedModel = value;
    if (modelSelect) modelSelect.value = value;
    selectedContext = selectedContext?.matchingPixels ? { ...selectedContext, model: value } : null;
    draw();
    return true;
  }

  function setUnavailable(message) {
    selectedContext = null;
    clearCanvases();
    caption.textContent = 'No paired raster figure is available for this selection.';
    detail.textContent = '';
    status.textContent = message;
    outcomeLegend.hidden = true;
    classLegend.hidden = false;
  }

  function recordFromUrl() {
    const requested = new URLSearchParams(location.search).get('public-case');
    return cases.find((item) => item.case_id === requested);
  }

  restore(element);
  if (caseGroup) fillCases(caseGroup.value || element.dataset.initialGroup, element.dataset.initialCase);
  if (modelSelect) selectedModel = modelSelect.value || selectedModel;
  if (layerSelect) selectedLayer = layerSelect.value || selectedLayer;
  if (swipe) element.style.setProperty('--split', swipe.value + '%');
  if (opacity) element.style.setProperty('--overlay-opacity', String(Number(opacity.value) / 100));
  if (zoom) updateTransform();
  const fromUrl = recordFromUrl();
  const initial = cases.find((item) => item.case_id === (fromUrl?.case_id ?? element.dataset.initialCase))
    ?? groupsFor(caseGroup?.value || element.dataset.initialGroup)[0];
  if (initial) setCase(initial);

  caseGroup?.addEventListener('change', () => {
    fillCases(caseGroup.value);
    const selected = cases.find((item) => item.case_id === caseSelect.value);
    if (selected) setCase(selected);
    share(element);
  });
  caseSelect?.addEventListener('change', () => {
    const selected = cases.find((item) => item.case_id === caseSelect.value);
    if (selected) setCase(selected);
    share(element);
  });
  modelSelect?.addEventListener('change', () => {
    setModel(modelSelect.value);
    share(element);
  });
  layerSelect?.addEventListener('change', () => {
    selectedLayer = layerSelect.value;
    draw();
    share(element);
  });
  swipe?.addEventListener('input', () => {
    element.style.setProperty('--split', swipe.value + '%');
    share(element);
  });
  opacity?.addEventListener('input', () => {
    element.style.setProperty('--overlay-opacity', String(Number(opacity.value) / 100));
    share(element);
  });
  zoom?.addEventListener('input', () => {
    panX = panY = 0;
    updateTransform();
    share(element);
  });
  viewport.addEventListener('pointerdown', (event) => {
    if (Number(zoom?.value || 1) <= 1 || event.button !== 0) return;
    drag = { x: event.clientX, y: event.clientY, panX, panY };
    viewport.setPointerCapture(event.pointerId);
  });
  viewport.addEventListener('pointermove', (event) => {
    if (!drag) return;
    const maxX = viewport.clientWidth * (Number(zoom.value) - 1) / 2;
    const maxY = viewport.clientHeight * (Number(zoom.value) - 1) / 2;
    panX = Math.max(-maxX, Math.min(maxX, drag.panX + event.clientX - drag.x));
    panY = Math.max(-maxY, Math.min(maxY, drag.panY + event.clientY - drag.y));
    updateTransform();
  });
  viewport.addEventListener('pointerup', () => { drag = null; });
  viewport.addEventListener('pointercancel', () => { drag = null; });
  viewport.addEventListener('keydown', (event) => {
    if (Number(zoom?.value || 1) <= 1) return;
    const amount = 24;
    if (event.key === 'ArrowLeft') panX += amount;
    else if (event.key === 'ArrowRight') panX -= amount;
    else if (event.key === 'ArrowUp') panY += amount;
    else if (event.key === 'ArrowDown') panY -= amount;
    else return;
    event.preventDefault();
    const maxX = viewport.clientWidth * (Number(zoom.value) - 1) / 2;
    const maxY = viewport.clientHeight * (Number(zoom.value) - 1) / 2;
    panX = Math.max(-maxX, Math.min(maxX, panX));
    panY = Math.max(-maxY, Math.min(maxY, panY));
    updateTransform();
  });

  const instance = { setCase, setModel, setUnavailable, selectedCase: () => selectedCase };
  instances.set(element, instance);
  return instance;
}

export function explorerFor(element) {
  return instances.get(element) ?? init(element);
}

document.querySelectorAll('[data-paired-explorer]').forEach(init);
