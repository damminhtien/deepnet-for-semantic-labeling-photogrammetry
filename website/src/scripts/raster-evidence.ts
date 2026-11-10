import manifest from '../generated/public-crops.json';
import { restore, share } from './state.mjs';
import { route } from './routes.mjs';
import { composeHistoricalMappings } from './spectral-mapping.mjs';

type Crop = (typeof manifest.assets)[number];
type Layer = 'prediction' | 'reference' | 'confusion' | 'error' | 'paired_outcome' | 'source' | 'infrared' | 'spectral_compare';

const assets = new Map<string, Crop>(manifest.assets.map((asset) => [asset.case_id, asset]));
const imageCache = new Map<string, Promise<HTMLImageElement>>();
const instances = new WeakMap<HTMLElement, RasterExplorer>();
const outcomeColors = ['#009e73', '#999999', '#e69f00', '#0072b2', '#d55e00'].map(parseColor);
const layerNames: Record<Layer, string> = {
  prediction: 'Prediction',
  reference: 'Eroded reference',
  confusion: 'Selected confusion pair',
  error: 'Model errors',
  paired_outcome: 'Paired outcome',
  source: 'RGB source',
  infrared: 'Infrared band · grayscale',
  spectral_compare: 'Observed vs implied historical bands',
};

function parseColor(value: string): [number, number, number] {
  const hex = value.replace('#', '');
  return [0, 2, 4].map((offset) => Number.parseInt(hex.slice(offset, offset + 2), 16)) as [number, number, number];
}

function loadImage(url: string): Promise<HTMLImageElement> {
  if (!imageCache.has(url)) {
    imageCache.set(url, new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error(`Could not load research crop: ${url}`));
      image.src = url;
    }));
  }
  return imageCache.get(url)!;
}

function cropAsset(url: string): string {
  return route() + url;
}

export class RasterExplorer {
  private readonly base: HTMLCanvasElement;
  private readonly overlay: HTMLCanvasElement;
  private readonly viewport: HTMLElement;
  private readonly fallback: HTMLImageElement;
  private readonly caseSelect: HTMLSelectElement | null;
  private readonly modelSelect: HTMLSelectElement | null;
  private readonly layerSelect: HTMLSelectElement | null;
  private readonly truthSelect: HTMLSelectElement | null;
  private readonly predictionSelect: HTMLSelectElement | null;
  private readonly swipe: HTMLInputElement;
  private readonly opacity: HTMLInputElement;
  private readonly zoom: HTMLInputElement;
  private readonly status: HTMLElement;
  private readonly caption: HTMLElement;
  private readonly detail: HTMLElement;
  private readonly provenance: HTMLElement;
  private readonly layerName: HTMLElement;
  private readonly sourceSize: HTMLElement;
  private readonly classLegend: HTMLElement;
  private readonly outcomeLegend: HTMLElement;
  private readonly spectralNote: HTMLElement;
  private readonly spectralIrrelevantControls: HTMLElement[];
  private selectedCase: Crop | null = null;
  private selectedModel = 'dinov3';
  private selectedLayer: Layer = 'prediction';
  private filterTruth: number | null = null;
  private filterPrediction: number | null = null;
  private ids: Uint8ClampedArray | null = null;
  private rgbPixels: Uint8ClampedArray | null = null;
  private infraredPixels: Uint8ClampedArray | null = null;
  private spectralObserved: Uint8ClampedArray | null = null;
  private spectralImplied: Uint8ClampedArray | null = null;
  private panX = 0;
  private panY = 0;
  private drag: { x: number; y: number; panX: number; panY: number } | null = null;

  constructor(private readonly root: HTMLElement) {
    this.base = root.querySelector<HTMLCanvasElement>('[data-base]')!;
    this.overlay = root.querySelector<HTMLCanvasElement>('[data-overlay]')!;
    this.viewport = root.querySelector<HTMLElement>('[data-viewport]')!;
    this.fallback = root.querySelector<HTMLImageElement>('[data-fallback]')!;
    this.caseSelect = root.querySelector<HTMLSelectElement>('[data-case]');
    this.modelSelect = root.querySelector<HTMLSelectElement>('[data-model]');
    this.layerSelect = root.querySelector<HTMLSelectElement>('[data-layer]')!;
    this.truthSelect = root.querySelector<HTMLSelectElement>('[data-truth-class]')!;
    this.predictionSelect = root.querySelector<HTMLSelectElement>('[data-predicted-class]')!;
    this.swipe = root.querySelector<HTMLInputElement>('[data-swipe]')!;
    this.opacity = root.querySelector<HTMLInputElement>('[data-opacity]')!;
    this.zoom = root.querySelector<HTMLInputElement>('[data-zoom]')!;
    this.status = root.querySelector<HTMLElement>('[data-status]')!;
    this.caption = root.querySelector<HTMLElement>('[data-case-caption]')!;
    this.detail = root.querySelector<HTMLElement>('[data-case-detail]')!;
    this.provenance = root.querySelector<HTMLElement>('[data-model-provenance]')!;
    this.layerName = root.querySelector<HTMLElement>('[data-layer-name]')!;
    this.sourceSize = root.querySelector<HTMLElement>('[data-source-size]')!;
    this.classLegend = root.querySelector<HTMLElement>('[data-class-legend]')!;
    this.outcomeLegend = root.querySelector<HTMLElement>('[data-outcome-legend]')!;
    this.spectralNote = root.querySelector<HTMLElement>('[data-spectral-note]')!;
    this.spectralIrrelevantControls = Array.from(root.querySelectorAll<HTMLElement>('[data-spectral-irrelevant]'));
    this.bind();
  }

  async setCase(record: string | { case_id?: string; scene?: string; scene_id?: string; row?: number; col?: number; pixel_window?: { row: number; col: number } }, context: { matchingPixels?: number; model?: string } | null = null): Promise<boolean> {
    const found = typeof record === 'string'
      ? assets.get(record)
      : assets.get(record.case_id ?? '') ?? manifest.assets.find((item) => {
        const window = record.pixel_window ?? record;
        return item.scene_id === (record.scene_id ?? record.scene)
          && item.pixel_window.row === (window.row ?? record.row)
          && item.pixel_window.col === (window.col ?? record.col);
      });
    if (!found) return false;
    this.selectedCase = found;
    this.ids = null;
    this.rgbPixels = null;
    this.infraredPixels = null;
    this.spectralObserved = null;
    this.spectralImplied = null;
    if (this.caseSelect) this.caseSelect.value = found.case_id;
    this.fallback.src = cropAsset(found.files.rgb.url);
    this.fallback.width = found.pixel_window.width;
    this.fallback.height = found.pixel_window.height;
    this.root.style.setProperty('--split', `${this.swipe.value}%`);
    this.root.dataset.ready = 'false';
    const urls = [found.files.rgb.url, found.files.infrared.url, found.files.class_ids.url].map(cropAsset);
    try {
      const [rgbImage, infraredImage, idsImage] = await Promise.all(urls.map(loadImage));
      if (this.selectedCase?.case_id !== found.case_id) return true;
      const { width, height } = found.pixel_window;
      this.base.width = this.overlay.width = width;
      this.base.height = this.overlay.height = height;
      this.viewport.style.aspectRatio = `${width} / ${height}`;
      const baseContext = this.base.getContext('2d', { willReadFrequently: true })!;
      baseContext.imageSmoothingEnabled = false;
      baseContext.clearRect(0, 0, width, height);
      baseContext.drawImage(rgbImage, 0, 0, width, height);
      this.rgbPixels = baseContext.getImageData(0, 0, width, height).data;
      const infraredCanvas = document.createElement('canvas');
      infraredCanvas.width = width;
      infraredCanvas.height = height;
      const infraredContext = infraredCanvas.getContext('2d', { willReadFrequently: true })!;
      infraredContext.drawImage(infraredImage, 0, 0, width, height);
      this.infraredPixels = infraredContext.getImageData(0, 0, width, height).data;
      const spectral = composeHistoricalMappings(this.rgbPixels, this.infraredPixels);
      this.spectralObserved = spectral.observed;
      this.spectralImplied = spectral.implied;
      const idsCanvas = document.createElement('canvas');
      idsCanvas.width = width;
      idsCanvas.height = height;
      const idsContext = idsCanvas.getContext('2d', { willReadFrequently: true })!;
      idsContext.imageSmoothingEnabled = false;
      idsContext.drawImage(idsImage, 0, 0, width, height);
      this.ids = idsContext.getImageData(0, 0, width, height).data;
      this.root.dataset.ready = 'true';
      this.root.dataset.unavailable = 'false';
      this.caption.textContent = `Scene ${found.scene_id} · ${found.case_group_title}`;
      const matching = context?.matchingPixels === undefined
        ? ''
        : ` · ${context.matchingPixels.toLocaleString('en-US')} selected ${context.model ?? this.selectedModel} pixels`;
      this.detail.textContent = `Rows ${found.pixel_window.row}–${found.pixel_window.row + height - 1}, columns ${found.pixel_window.col}–${found.pixel_window.col + width - 1} · ${found.pixel_window.valid_pixels.toLocaleString('en-US')} scored pixels${matching}`;
      this.provenance.textContent = (['dinov3', 'pspnet'] as const).map((name) => `${found.models[name].model_id} · SHA-256 ${found.models[name].checkpoint_sha256.slice(0, 12)}`).join(' / ');
      this.sourceSize.textContent = `${width} × ${height} source pixels · 5 cm GSD`;
      this.draw();
      if (this.root.dataset.context === 'Evaluation') {
        const url = new URL(location.href);
        url.searchParams.set('raster-case', found.case_id);
        history.replaceState(null, '', url);
      }
      return true;
    } catch (error) {
      this.unavailable(error instanceof Error ? error.message : 'Could not load this recorded crop.');
      return false;
    }
  }

  setModel(model: string): boolean {
    if (!['dinov3', 'pspnet'].includes(model)) return false;
    this.selectedModel = model;
    if (this.modelSelect) this.modelSelect.value = model;
    this.draw();
    return true;
  }

  setConfusion(truth: number, prediction: number): void {
    this.filterTruth = truth;
    this.filterPrediction = prediction;
    if (this.truthSelect) this.truthSelect.value = String(truth);
    if (this.predictionSelect) this.predictionSelect.value = String(prediction);
    this.selectedLayer = 'confusion';
    if (this.layerSelect) this.layerSelect.value = this.selectedLayer;
    this.draw();
  }

  setUnavailable(message: string): void {
    this.unavailable(message);
  }

  private unavailable(message: string): void {
    this.ids = null;
    this.root.dataset.ready = 'false';
    this.root.dataset.unavailable = 'true';
    this.caption.textContent = 'No pixel-aligned raster is available for this selection.';
    this.detail.textContent = '';
    this.provenance.textContent = '';
    this.status.textContent = message;
  }

  private bind(): void {
    restore(this.root);
    if (this.modelSelect) this.selectedModel = this.modelSelect.value || this.selectedModel;
    this.selectedLayer = (this.layerSelect?.value || this.selectedLayer) as Layer;
    this.root.style.setProperty('--split', `${this.swipe.value}%`);
    this.root.style.setProperty('--overlay-opacity', String(Number(this.opacity.value) / 100));
    this.caseSelect?.addEventListener('change', () => {
      void this.setCase(this.caseSelect!.value);
      share(this.root);
    });
    this.modelSelect?.addEventListener('change', () => {
      this.setModel(this.modelSelect!.value);
      share(this.root);
    });
    const layerSelect = this.layerSelect;
    layerSelect?.addEventListener('change', () => {
      this.selectedLayer = layerSelect.value as Layer;
      this.draw();
      share(this.root);
    });
    const truthSelect = this.truthSelect;
    truthSelect?.addEventListener('change', () => {
      this.filterTruth = truthSelect.value === 'all' ? null : Number(truthSelect.value);
      if (this.selectedLayer === 'confusion') this.draw();
      share(this.root);
    });
    const predictionSelect = this.predictionSelect;
    predictionSelect?.addEventListener('change', () => {
      this.filterPrediction = predictionSelect.value === 'all' ? null : Number(predictionSelect.value);
      if (this.selectedLayer === 'confusion') this.draw();
      share(this.root);
    });
    this.swipe.addEventListener('input', () => {
      this.root.style.setProperty('--split', `${this.swipe.value}%`);
      share(this.root);
    });
    this.opacity.addEventListener('input', () => {
      this.root.style.setProperty('--overlay-opacity', String(Number(this.opacity.value) / 100));
      share(this.root);
    });
    this.zoom.addEventListener('input', () => {
      this.panX = this.panY = 0;
      this.updateTransform();
      share(this.root);
    });
    this.viewport.addEventListener('pointerdown', (event) => {
      const pointer = event as PointerEvent;
      if (Number(this.zoom.value) <= 1 || pointer.button !== 0) return;
      this.drag = { x: pointer.clientX, y: pointer.clientY, panX: this.panX, panY: this.panY };
      this.viewport.setPointerCapture(pointer.pointerId);
    });
    this.viewport.addEventListener('pointermove', (event) => {
      const pointer = event as PointerEvent;
      if (this.drag) {
        const maxX = this.viewport.clientWidth * (Number(this.zoom.value) - 1) / 2;
        const maxY = this.viewport.clientHeight * (Number(this.zoom.value) - 1) / 2;
        this.panX = Math.max(-maxX, Math.min(maxX, this.drag.panX + pointer.clientX - this.drag.x));
        this.panY = Math.max(-maxY, Math.min(maxY, this.drag.panY + pointer.clientY - this.drag.y));
        this.updateTransform();
      }
      this.inspectPixel(pointer);
    });
    this.viewport.addEventListener('pointerleave', () => {
      if (!this.drag) this.resetStatus();
    });
    this.viewport.addEventListener('pointerup', () => { this.drag = null; });
    this.viewport.addEventListener('pointercancel', () => { this.drag = null; });
    this.viewport.addEventListener('keydown', (event) => {
      if (Number(this.zoom.value) <= 1) return;
      const key = event as KeyboardEvent;
      const amount = 24;
      if (key.key === 'ArrowLeft') this.panX += amount;
      else if (key.key === 'ArrowRight') this.panX -= amount;
      else if (key.key === 'ArrowUp') this.panY += amount;
      else if (key.key === 'ArrowDown') this.panY -= amount;
      else return;
      key.preventDefault();
      const maxX = this.viewport.clientWidth * (Number(this.zoom.value) - 1) / 2;
      const maxY = this.viewport.clientHeight * (Number(this.zoom.value) - 1) / 2;
      this.panX = Math.max(-maxX, Math.min(maxX, this.panX));
      this.panY = Math.max(-maxY, Math.min(maxY, this.panY));
      this.updateTransform();
    });

    const urlCase = new URLSearchParams(location.search).get('raster-case');
    const initialCase = assets.has(urlCase ?? '') ? urlCase! : this.root.dataset.initialCase!;
    void this.setCase(initialCase);
    this.draw();
  }

  private draw(): void {
    const crop = this.selectedCase;
    if (!crop || !this.ids) return;
    const width = crop.pixel_window.width;
    const height = crop.pixel_window.height;
    const baseContext = this.base.getContext('2d', { willReadFrequently: true })!;
    baseContext.imageSmoothingEnabled = false;
    baseContext.clearRect(0, 0, width, height);
    if (this.selectedLayer === 'infrared' && this.infraredPixels) {
      const ir = new ImageData(new Uint8ClampedArray(this.infraredPixels), width, height);
      for (let offset = 0; offset < ir.data.length; offset += 4) {
        const gray = ir.data[offset];
        ir.data[offset + 1] = gray;
        ir.data[offset + 2] = gray;
      }
      baseContext.putImageData(ir, 0, 0);
    } else if (this.selectedLayer === 'spectral_compare' && this.spectralObserved) {
      baseContext.putImageData(new ImageData(new Uint8ClampedArray(this.spectralObserved), width, height), 0, 0);
    } else if (this.rgbPixels) {
      baseContext.putImageData(new ImageData(new Uint8ClampedArray(this.rgbPixels), width, height), 0, 0);
    }
    this.drawOverlay();
    this.layerName.textContent = layerNames[this.selectedLayer];
    this.outcomeLegend.hidden = this.selectedLayer !== 'paired_outcome';
    this.classLegend.hidden = ['paired_outcome', 'source', 'infrared', 'spectral_compare'].includes(this.selectedLayer);
    this.spectralNote.hidden = this.selectedLayer !== 'spectral_compare';
    for (const control of this.spectralIrrelevantControls) control.hidden = this.selectedLayer === 'spectral_compare';
    this.resetStatus();
    this.root.style.setProperty('--split', `${this.swipe.value}%`);
    this.root.style.setProperty('--overlay-opacity', String(Number(this.opacity.value) / 100));
    this.updateTransform();
  }

  private drawOverlay(): void {
    const crop = this.selectedCase;
    if (!crop || !this.ids) return;
    const width = crop.pixel_window.width;
    const height = crop.pixel_window.height;
    const context = this.overlay.getContext('2d')!;
    const output = context.createImageData(width, height);
    const colors = crop.class_colors as number[][];
    const selectedPredIndex = this.selectedModel === 'dinov3' ? 1 : 2;
    const alpha = 232;
    const isSource = this.selectedLayer === 'source' || this.selectedLayer === 'infrared';
    if (isSource) {
      this.overlay.style.visibility = 'hidden';
      return;
    }
    this.overlay.style.visibility = 'visible';
    if (this.selectedLayer === 'spectral_compare') {
      if (this.spectralImplied) {
        context.putImageData(new ImageData(new Uint8ClampedArray(this.spectralImplied), width, height), 0, 0);
      } else {
        context.clearRect(0, 0, width, height);
      }
      return;
    }
    for (let pixel = 0; pixel < width * height; pixel += 1) {
      const sourceOffset = pixel * 4;
      const targetOffset = sourceOffset;
      const truth = this.ids[sourceOffset];
      const dino = this.ids[sourceOffset + 1];
      const psp = this.ids[sourceOffset + 2];
      const prediction = selectedPredIndex === 1 ? dino : psp;
      let color: number[] | null = null;
      let opacity = alpha;
      if (this.selectedLayer === 'reference') {
        if (truth < 6) color = colors[truth];
      } else if (this.selectedLayer === 'prediction') {
        if (truth < 6) color = colors[prediction];
      } else if (this.selectedLayer === 'confusion') {
        if (truth < 6 && prediction < 6
          && (this.filterTruth === null || truth === this.filterTruth)
          && (this.filterPrediction === null || prediction === this.filterPrediction)) {
          color = [230, 159, 0];
          opacity = 245;
        }
      } else if (this.selectedLayer === 'error') {
        if (truth < 6 && prediction !== truth) color = [213, 94, 0];
      } else if (this.selectedLayer === 'paired_outcome' && truth < 6) {
        const dinoCorrect = dino === truth;
        const pspCorrect = psp === truth;
        const index = dinoCorrect && pspCorrect ? 0
          : !dinoCorrect && !pspCorrect && dino === psp ? 1
          : !dinoCorrect && !pspCorrect ? 2
          : dinoCorrect ? 3 : 4;
        color = outcomeColors[index];
      }
      if (color) {
        output.data[targetOffset] = color[0];
        output.data[targetOffset + 1] = color[1];
        output.data[targetOffset + 2] = color[2];
        output.data[targetOffset + 3] = opacity;
      }
    }
    context.putImageData(output, 0, 0);
  }

  private inspectPixel(event: PointerEvent): void {
    const crop = this.selectedCase;
    if (!crop || !this.ids || event.buttons === 1) return;
    const rect = this.base.getBoundingClientRect();
    const x = Math.floor((event.clientX - rect.left) * crop.pixel_window.width / rect.width);
    const y = Math.floor((event.clientY - rect.top) * crop.pixel_window.height / rect.height);
    if (x < 0 || y < 0 || x >= crop.pixel_window.width || y >= crop.pixel_window.height) return;
    const offset = (y * crop.pixel_window.width + x) * 4;
    const [truth, dino, psp] = [this.ids[offset], this.ids[offset + 1], this.ids[offset + 2]];
    const coordinate = `r${crop.pixel_window.row + y}, c${crop.pixel_window.col + x}`;
    const opticalOffset = offset;
    const red = this.rgbPixels?.[opticalOffset] ?? 0;
    const green = this.rgbPixels?.[opticalOffset + 1] ?? 0;
    const blue = this.rgbPixels?.[opticalOffset + 2] ?? 0;
    const infrared = this.infraredPixels?.[opticalOffset] ?? 0;
    if (this.selectedLayer === 'spectral_compare') {
      this.status.textContent = `${coordinate} · source bands B1/B2/B3/B4 ${red}/${green}/${blue}/${infrared} DN · left B4/B2/B3 · right B4/B1/B2.`;
      return;
    }
    if (truth === manifest.ignore_class_id) {
      this.status.textContent = `${coordinate} · RGB ${red}/${green}/${blue} · IR ${infrared} DN · ignored by the eroded-reference protocol.`;
      return;
    }
    const names = manifest.class_names;
    this.status.textContent = `${coordinate} · RGB ${red}/${green}/${blue} · IR ${infrared} DN · reference: ${names[truth]} · DINOv3: ${names[dino]} · PSPNet: ${names[psp]}`;
  }

  private resetStatus(): void {
    this.status.textContent = this.selectedLayer === 'spectral_compare'
      ? 'Move over the image to inspect source coordinates, B1–B4 values, and both channel mappings.'
      : 'Move over the image to inspect source coordinates and the three aligned class IDs.';
  }

  private updateTransform(): void {
    const factor = Number(this.zoom.value) || 1;
    const value = `translate(${this.panX}px, ${this.panY}px) scale(${factor})`;
    this.base.style.transform = value;
    this.overlay.style.transform = value;
    this.viewport.dataset.zoomed = String(factor > 1);
    if (factor === 1) this.panX = this.panY = 0;
  }
}

document.querySelectorAll<HTMLElement>('[data-raster-explorer]').forEach((root) => {
  if (!instances.has(root)) instances.set(root, new RasterExplorer(root));
});

export function rasterExplorerFor(root: HTMLElement): RasterExplorer {
  const existing = instances.get(root);
  if (existing) return existing;
  const explorer = new RasterExplorer(root);
  instances.set(root, explorer);
  return explorer;
}
